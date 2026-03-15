#include "hf_llama_loader.h"

#include <fstream>
#include <sstream>

#include <filesystem>
#include <nlohmann/json.hpp>
namespace fs = std::filesystem;

#include "../transformer/llama_model.h"
#include "safetensors_loader.h"

namespace serius {
namespace io {

namespace {

model::LLamaConfig parse_llama_config(const nlohmann::json &j)
{
    model::LLamaConfig c;
    c.vocab_size = j.value("vocab_size", c.vocab_size);
    c.hidden_size = j.value("hidden_size", c.hidden_size);
    c.num_attention_heads = j.value("num_attention_heads", c.num_attention_heads);
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.num_hidden_layers = j.value("num_hidden_layers", j.value("num_layers", c.num_hidden_layers));
    c.intermediate_size = j.value("intermediate_size", c.intermediate_size);
    c.max_position_embeddings = j.value("max_position_embeddings", c.max_position_embeddings);
    c.rms_norm_eps = j.value("rms_norm_eps", c.rms_norm_eps);
    c.rope_theta = j.value("rope_theta", c.rope_theta);
    return c;
}

void transpose_hf_to_cpp(const float *hf, float *cpp, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) cpp[j * rows + i] = hf[i * cols + j];
}

bool copy_into_param(SafetensorsFile &sf, const std::string &name, float *dest, size_t size,
                     bool transpose_weights)
{
    auto it = sf.tensors().find(name);
    if (it == sf.tensors().end()) return false;
    const auto &m = it->second;
    size_t num = 1;
    for (int64_t d : m.shape) num *= static_cast<size_t>(d);
    if (num != size) return false;
    if (!transpose_weights)
        return sf.copy_tensor_to_float(name, dest, size);
    std::vector<float> tmp(size);
    if (!sf.copy_tensor_to_float(name, tmp.data(), size)) return false;
    size_t rows = static_cast<size_t>(m.shape[0]);
    size_t cols = static_cast<size_t>(m.shape[1]);
    transpose_hf_to_cpp(tmp.data(), dest, rows, cols);
    return true;
}

bool load_one_file_into_model(SafetensorsFile &sf, model::LLaMAModel *model)
{
    const model::LLamaConfig &c = model->config();
    size_t hidden = c.hidden_size;
    size_t n_heads = c.num_attention_heads;
    size_t n_kv = c.num_key_value_heads;
    size_t head_dim = hidden / n_heads;
    size_t n_q = n_heads * head_dim;
    size_t n_kv_dim = n_kv * head_dim;
    size_t inter = c.intermediate_size;
    size_t vocab = c.vocab_size;

    if (sf.tensors().count("model.embed_tokens.weight"))
        copy_into_param(sf, "model.embed_tokens.weight", model->embed_tokens()->data.data(),
                        vocab * hidden, false);

    for (size_t i = 0; i < c.num_hidden_layers; ++i) {
        model::LLaMABlock *block = model->layers()[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        copy_into_param(sf, prefix + "input_layernorm.weight",
                        block->input_layernorm_weight->data.data(), hidden, false);
        copy_into_param(sf, prefix + "self_attn.q_proj.weight", block->q_proj->data.data(),
                        hidden * n_q, true);
        copy_into_param(sf, prefix + "self_attn.k_proj.weight", block->k_proj->data.data(),
                        hidden * n_kv_dim, true);
        copy_into_param(sf, prefix + "self_attn.v_proj.weight", block->v_proj->data.data(),
                        hidden * n_kv_dim, true);
        copy_into_param(sf, prefix + "self_attn.o_proj.weight", block->o_proj->data.data(),
                        n_q * hidden, true);
        copy_into_param(sf, prefix + "post_attention_layernorm.weight",
                        block->post_attention_layernorm_weight->data.data(), hidden, false);
        copy_into_param(sf, prefix + "mlp.gate_proj.weight", block->gate_proj->data.data(),
                        hidden * inter, true);
        copy_into_param(sf, prefix + "mlp.up_proj.weight", block->up_proj->data.data(),
                        hidden * inter, true);
        copy_into_param(sf, prefix + "mlp.down_proj.weight", block->down_proj->data.data(),
                        inter * hidden, true);
    }

    if (sf.tensors().count("model.norm.weight"))
        copy_into_param(sf, "model.norm.weight", model->norm_weight()->data.data(), hidden, false);
    if (sf.tensors().count("lm_head.weight"))
        copy_into_param(sf, "lm_head.weight", model->lm_head()->data.data(), vocab * hidden, true);  // HF (vocab,hidden) -> (hidden,vocab)
    else if (sf.tensors().count("model.embed_tokens.weight"))
        copy_into_param(sf, "model.embed_tokens.weight", model->lm_head()->data.data(), vocab * hidden,
                        true);
    return true;
}

}  // namespace

model::LLamaConfig load_llama_config(const std::string &config_path)
{
    std::ifstream f(config_path);
    if (!f) return model::LLamaConfig{};
    std::stringstream ss;
    ss << f.rdbuf();
    try {
        auto j = nlohmann::json::parse(ss.str());
        return parse_llama_config(j);
    } catch (const std::exception &) {
        return model::LLamaConfig{};
    }
}

bool load_llama_weights_from_file(const std::string &safetensors_path, model::LLaMAModel *model)
{
    if (!model) return false;
    SafetensorsFile sf;
    if (!load_safetensors(safetensors_path, &sf)) return false;
    return load_one_file_into_model(sf, model);
}

bool load_llama_weights_from_dir(const std::string &dir_path, model::LLaMAModel *model)
{
    if (!model) return false;
    bool any = false;
    try {
        for (const auto &entry : fs::directory_iterator(dir_path)) {
            if (!entry.is_regular_file()) continue;
            std::string path = entry.path().string();
            if (path.size() < 12) continue;
            if (path.compare(path.size() - 12, 12, ".safetensors") != 0) continue;
            SafetensorsFile sf;
            if (!load_safetensors(path, &sf)) continue;
            any = true;
            load_one_file_into_model(sf, model);
        }
    } catch (const std::exception &) {
    }
    return any;
}

}  // namespace io
}  // namespace serius
