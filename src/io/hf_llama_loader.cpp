#include "hf_llama_loader.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <vector>
#include <string>

#include <nlohmann/json.hpp>

#include "../transformer/llama_model.h"
#include "safetensors_loader.h"

namespace fs = std::filesystem;

namespace serius {
namespace io {

namespace {

/**
 * @brief Parses the LLaMA configuration from a JSON object.
 */
model::LLamaConfig parse_llama_config(const nlohmann::json &json_data)
{
    model::LLamaConfig llama_cfg;
    llama_cfg.vocab_size = json_data.value("vocab_size", llama_cfg.vocab_size);
    llama_cfg.hidden_size = json_data.value("hidden_size", llama_cfg.hidden_size);
    llama_cfg.num_attention_heads = json_data.value("num_attention_heads", llama_cfg.num_attention_heads);
    llama_cfg.num_key_value_heads = json_data.value("num_key_value_heads", llama_cfg.num_attention_heads);
    llama_cfg.num_hidden_layers = json_data.value("num_hidden_layers", json_data.value("num_layers", llama_cfg.num_hidden_layers));
    llama_cfg.intermediate_size = json_data.value("intermediate_size", llama_cfg.intermediate_size);
    llama_cfg.max_position_embeddings = json_data.value("max_position_embeddings", llama_cfg.max_position_embeddings);
    llama_cfg.rms_norm_eps = json_data.value("rms_norm_eps", llama_cfg.rms_norm_eps);
    llama_cfg.rope_theta = json_data.value("rope_theta", llama_cfg.rope_theta);
    return llama_cfg;
}

/**
 * @brief Transposes a matrix from HuggingFace format to the format used by the C++ engine.
 */
void transpose_hf_to_cpp(const float *src_buffer, float *dst_buffer, size_t num_rows, size_t num_cols)
{
    for (size_t r_idx = 0; r_idx < num_rows; ++r_idx) {
        for (size_t c_idx = 0; c_idx < num_cols; ++c_idx) {
            dst_buffer[(c_idx * num_rows) + r_idx] = src_buffer[(r_idx * num_cols) + c_idx];
        }
    }
}

/**
 * @brief Copies a single parameter from a safetensors file into a destination buffer.
 */
bool copy_into_param(SafetensorsFile &safetensors_file, const std::string &param_name, float *destination, size_t expected_size,
                     bool should_transpose)
{
    auto found_tensor = safetensors_file.tensors().find(param_name);
    if (found_tensor == safetensors_file.tensors().end()) {
        return false;
    }
    const auto &metadata = found_tensor->second;
    size_t actual_element_count = 1;
    for (int64_t dim : metadata.shape) {
        actual_element_count *= static_cast<size_t>(dim);
    }
    if (actual_element_count != expected_size) {
        return false;
    }
    if (!should_transpose) {
        return safetensors_file.copy_tensor_to_float(param_name, destination, expected_size);
    }
    std::vector<float> tmp_buffer(expected_size);
    if (!safetensors_file.copy_tensor_to_float(param_name, tmp_buffer.data(), expected_size)) {
        return false;
    }
    size_t rows = static_cast<size_t>(metadata.shape[0]);
    size_t cols = static_cast<size_t>(metadata.shape[1]);
    transpose_hf_to_cpp(tmp_buffer.data(), destination, rows, cols);
    return true;
}

/**
 * @brief Copies a parameter with Tensor Parallel (TP) splitting.
 */
bool copy_into_param_tp(SafetensorsFile &safetensors_file, const std::string &param_name, float *destination, size_t expected_local_size,
                        bool should_transpose_after_slice, int current_tp_rank, int total_tp_size, bool is_column_parallel)
{
    if (total_tp_size <= 1) {
        return copy_into_param(safetensors_file, param_name, destination, expected_local_size, should_transpose_after_slice);
    }

    auto found_tensor = safetensors_file.tensors().find(param_name);
    if (found_tensor == safetensors_file.tensors().end()) {
        return false;
    }
    const auto &metadata = found_tensor->second;
    if (metadata.shape.size() != 2) {
        return false;
    }
    
    size_t rows = static_cast<size_t>(metadata.shape[0]);
    size_t cols = static_cast<size_t>(metadata.shape[1]);
    size_t full_size = rows * cols;
    if (full_size == 0) {
        return false;
    }

    std::vector<float> tmp_full(full_size);
    if (!safetensors_file.copy_tensor_to_float(param_name, tmp_full.data(), full_size)) {
        return false;
    }

    std::vector<float> math_full(full_size);
    if (should_transpose_after_slice) {
        transpose_hf_to_cpp(tmp_full.data(), math_full.data(), rows, cols);
        std::swap(rows, cols);
    } else {
        math_full = std::move(tmp_full);
    }

    size_t local_rows = is_column_parallel ? rows : (rows / static_cast<size_t>(total_tp_size));
    size_t local_cols = is_column_parallel ? (cols / static_cast<size_t>(total_tp_size)) : cols;
    if (expected_local_size != local_rows * local_cols) {
        return false;
    }

    size_t r_start = is_column_parallel ? 0 : (static_cast<size_t>(current_tp_rank) * local_rows);
    size_t c_start = is_column_parallel ? (static_cast<size_t>(current_tp_rank) * local_cols) : 0;

    for (size_t r_idx = 0; r_idx < local_rows; ++r_idx) {
        for (size_t c_idx = 0; c_idx < local_cols; ++c_idx) {
            destination[(r_idx * local_cols) + c_idx] = math_full[((r_start + r_idx) * cols) + (c_start + c_idx)];
        }
    }
    return true;
}

/**
 * @brief Helper to load a range of layers for the current pipeline stage.
 */
void load_layers_in_range(SafetensorsFile &safetensors_file, model::LLaMAModel *model, size_t start, size_t end)
{
    const model::LLamaConfig &cfg = model->config();
    const auto &topo = model->topology();
    const int tp_rank = topo.tp_rank;
    const int tp_size = topo.tp_size > 0 ? topo.tp_size : 1;
    const size_t tp_size_sz = static_cast<size_t>(tp_size);

    const size_t hidden = cfg.hidden_size;
    const size_t head_dim = hidden / cfg.num_attention_heads;
    const size_t local_q_dim = (cfg.num_attention_heads * head_dim) / tp_size_sz;
    const size_t local_kv_dim = (cfg.num_key_value_heads * head_dim) / tp_size_sz;
    const size_t local_inter = cfg.intermediate_size / tp_size_sz;

    for (size_t l_idx = start; l_idx < end; ++l_idx) {
        size_t local_idx = l_idx - start;
        model::LLaMABlock *block = model->layers()[local_idx].get();
        std::string layer_prefix = "model.layers." + std::to_string(l_idx) + ".";

        copy_into_param(safetensors_file, layer_prefix + "input_layernorm.weight", block->input_layernorm_weight->data.data(), hidden, false);
        copy_into_param_tp(safetensors_file, layer_prefix + "self_attn.q_proj.weight", block->q_proj->data.data(), hidden * local_q_dim, true, tp_rank, tp_size, true);
        copy_into_param_tp(safetensors_file, layer_prefix + "self_attn.k_proj.weight", block->k_proj->data.data(), hidden * local_kv_dim, true, tp_rank, tp_size, true);
        copy_into_param_tp(safetensors_file, layer_prefix + "self_attn.v_proj.weight", block->v_proj->data.data(), hidden * local_kv_dim, true, tp_rank, tp_size, true);
        copy_into_param_tp(safetensors_file, layer_prefix + "self_attn.o_proj.weight", block->o_proj->data.data(), local_q_dim * hidden, true, tp_rank, tp_size, false);
        copy_into_param(safetensors_file, layer_prefix + "post_attention_layernorm.weight", block->post_attention_layernorm_weight->data.data(), hidden, false);
        copy_into_param_tp(safetensors_file, layer_prefix + "mlp.gate_proj.weight", block->gate_proj->data.data(), hidden * local_inter, true, tp_rank, tp_size, true);
        copy_into_param_tp(safetensors_file, layer_prefix + "mlp.up_proj.weight", block->up_proj->data.data(), hidden * local_inter, true, tp_rank, tp_size, true);
        copy_into_param_tp(safetensors_file, layer_prefix + "mlp.down_proj.weight", block->down_proj->data.data(), local_inter * hidden, true, tp_rank, tp_size, false);
    }
}

/**
 * @brief Loads tensors from a single safetensors file into the model.
 */
bool load_one_file_into_model(SafetensorsFile &safetensors_file, model::LLaMAModel *model)
{
    const model::LLamaConfig &cfg = model->config();
    const auto &topo = model->topology();
    const int pp_rank = topo.pp_rank;
    const int pp_size = topo.pp_size > 0 ? topo.pp_size : 1;
    const size_t layers_per_stage = cfg.num_hidden_layers / static_cast<size_t>(pp_size);
    const size_t start_layer = static_cast<size_t>(pp_rank) * layers_per_stage;
    size_t end_layer = start_layer + layers_per_stage;
    if (pp_rank == pp_size - 1) { end_layer = cfg.num_hidden_layers; }

    // Emeddings (Stage 0)
    if (model->embed_tokens() != nullptr && safetensors_file.tensors().count("model.embed_tokens.weight") > 0) {
        copy_into_param(safetensors_file, "model.embed_tokens.weight", model->embed_tokens()->data.data(), cfg.vocab_size * cfg.hidden_size, false);
    }

    // Layers (Current PP stage)
    load_layers_in_range(safetensors_file, model, start_layer, end_layer);

    // Final Norm & Head (Last stage)
    if (model->norm_weight() != nullptr && safetensors_file.tensors().count("model.norm.weight") > 0) {
        copy_into_param(safetensors_file, "model.norm.weight", model->norm_weight()->data.data(), cfg.hidden_size, false);
    }
    
    if (model->lm_head() != nullptr) {
        if (safetensors_file.tensors().count("lm_head.weight") > 0) {
            copy_into_param(safetensors_file, "lm_head.weight", model->lm_head()->data.data(), cfg.vocab_size * cfg.hidden_size, true);
        } else if (safetensors_file.tensors().count("model.embed_tokens.weight") > 0) {
            copy_into_param(safetensors_file, "model.embed_tokens.weight", model->lm_head()->data.data(), cfg.vocab_size * cfg.hidden_size, true);
        }
    }
    return true;
}

}  // namespace

model::LLamaConfig load_llama_config(const std::string &config_path)
{
    std::ifstream config_file(config_path);
    if (!config_file) { return model::LLamaConfig{}; }
    std::stringstream buffer;
    buffer << config_file.rdbuf();
    try {
        auto json_data = nlohmann::json::parse(buffer.str());
        return parse_llama_config(json_data);
    } catch (const std::exception &) {
        return model::LLamaConfig{};
    }
}

bool load_llama_weights_from_file(const std::string &safetensors_path, model::LLaMAModel *model)
{
    if (model == nullptr) { return false; }
    SafetensorsFile safetensors_file;
    if (!load_safetensors(safetensors_path, &safetensors_file)) { return false; }
    return load_one_file_into_model(safetensors_file, model);
}

bool load_llama_weights_from_dir(const std::string &dir_path, model::LLaMAModel *model)
{
    if (model == nullptr) { return false; }
    bool any_file_loaded = false;
    try {
        for (const auto &entry : fs::directory_iterator(dir_path)) {
            if (!entry.is_regular_file()) { continue; }
            std::string path_str = entry.path().string();
            if (path_str.size() < 12 || path_str.compare(path_str.size() - 12, 12, ".safetensors") != 0) { continue; }
            SafetensorsFile safetensors_file;
            if (load_safetensors(path_str, &safetensors_file)) {
                any_file_loaded = true;
                load_one_file_into_model(safetensors_file, model);
            }
        }
    } catch (const std::exception &) {
        // Silently ignore DIR errors as some parts might be missing
    }
    return any_file_loaded;
}

}  // namespace io
}  // namespace serius
