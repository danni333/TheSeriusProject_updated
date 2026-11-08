#include "hf_llama_loader.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>

#include <nlohmann/json.hpp>

#include "../transformer/llama_model.h"
#include "safetensors_loader.h"
#include "gguf_loader.h"

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
 * @brief GGUF (llama arch) stores attn_q/attn_k output dims permuted for RoPE:
 * within each head, order is [even indices | odd indices] of the HF
 * head_dim layout, while HF uses plain [0..head_dim). This reorders each
 * row's out-features back to HF ordering (inverse of the llama.cpp
 * convert-time permute: reshape (n_head, head_dim/2, 2, n_embd) then
 * transpose(0,2,1,3)).
 * Buffer layout: [in_dim][out_dim] row-major; out_dim = n_heads * head_dim.
 */
void unpermute_gguf_rope_heads(float *weights, size_t in_dim, size_t out_dim, size_t head_dim)
{
    if (head_dim < 2 || (head_dim % 2) != 0 || out_dim % head_dim != 0) {
        return;
    }
    const size_t half = head_dim / 2;
    const size_t n_heads = out_dim / head_dim;
    std::vector<float> tmp(head_dim);
    for (size_t i = 0; i < in_dim; ++i) {
        float *row = weights + (i * out_dim);
        for (size_t h = 0; h < n_heads; ++h) {
            float *seg = row + (h * head_dim);
            for (size_t s = 0; s < head_dim; ++s) {
                // HF position s <- GGUF slot: evens at s/2, odds at (s-1)/2 + half
                const size_t src = ((s & 1) == 0) ? (s / 2) : ((s - 1) / 2 + half);
                tmp[s] = seg[src];
            }
            std::copy(tmp.begin(), tmp.end(), seg);
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
        copy_into_param(safetensors_file, layer_prefix + "self_attn.q_proj.bias", block->q_bias->data.data(), local_q_dim, false);
        copy_into_param(safetensors_file, layer_prefix + "self_attn.k_proj.bias", block->k_bias->data.data(), local_kv_dim, false);
        copy_into_param(safetensors_file, layer_prefix + "self_attn.v_proj.bias", block->v_bias->data.data(), local_kv_dim, false);
        copy_into_param_tp(safetensors_file, layer_prefix + "self_attn.o_proj.weight", block->o_proj->data.data(), local_q_dim * hidden, true, tp_rank, tp_size, false);
        copy_into_param(safetensors_file, layer_prefix + "post_attention_layernorm.weight", block->post_attention_layernorm_weight->data.data(), hidden, false);
        copy_into_param_tp(safetensors_file, layer_prefix + "mlp.gate_proj.weight", block->gate_proj->data.data(), hidden * local_inter, true, tp_rank, tp_size, true);
        copy_into_param_tp(safetensors_file, layer_prefix + "mlp.up_proj.weight", block->up_proj->data.data(), hidden * local_inter, true, tp_rank, tp_size, true);
        copy_into_param_tp(safetensors_file, layer_prefix + "mlp.down_proj.weight", block->down_proj->data.data(), local_inter * hidden, true, tp_rank, tp_size, false);
    }
}

/**
 * @brief Builds the transposed lm_head [vocab, hidden] used by the fast
 * row-major GEMV decode path. lm_head param layout is [hidden, vocab].
 */
void build_lm_head_t(model::LLaMAModel *model)
{
    if (model->lm_head() == nullptr) { return; }
    const size_t vocab  = model->config().vocab_size;
    const size_t hidden = model->config().hidden_size;
    core::Tensor &dst = model->lm_head_t();
    dst = core::Tensor({vocab, hidden});
    const float *src = model->lm_head()->data.data();
    float *d = dst.data();
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < static_cast<int64_t>(vocab); ++j) {
        const size_t jj = static_cast<size_t>(j);
        for (size_t i = 0; i < hidden; ++i) {
            d[jj * hidden + i] = src[i * vocab + jj];
        }
    }
}

// fp32 -> fp16 (round-to-nearest would be nicer; truncation error <0.1% is
// fine for block scales).
inline uint16_t f32_to_f16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = static_cast<int>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) { return static_cast<uint16_t>(sign); }
    if (exp >= 31) { return static_cast<uint16_t>(sign | 0x7C00u); }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

/**
 * @brief Quantizes an [N, K] row-major fp32 matrix into ggml-style Q4_0 blocks
 * (fp16 scale + 32 x 4-bit, w = (nib - 8) * d). Used to shrink large non-Q4
 * tensors such as a Q8_0 token_embd reused as lm_head.
 */
quant::QuantWeight quantize_fp32_to_q40(const float *w, size_t N, size_t K)
{
    quant::QuantWeight q;
    q.type = quant::QuantType::Q4_0;
    q.in = K;
    q.out = N;
    const size_t bpr = (K + 31) / 32;
    q.bytes.resize(N * bpr * 18);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < static_cast<int64_t>(N); ++r) {
        uint8_t *row = q.bytes.data() + static_cast<size_t>(r) * bpr * 18;
        const float *src = w + static_cast<size_t>(r) * K;
        for (size_t b = 0; b < bpr; ++b) {
            float amax = 0.0F;
            for (size_t e = 0; e < 32; ++e) {
                const size_t idx = b * 32 + e;
                const float v = (idx < K) ? src[idx] : 0.0F;
                amax = std::max(amax, std::fabs(v));
            }
            const float d = (amax > 0.0F) ? (amax / 7.0F) : 1e-8F;
            uint8_t *blk = row + b * 18;
            blk[0] = static_cast<uint8_t>(f32_to_f16(d) & 0xFF);
            blk[1] = static_cast<uint8_t>(f32_to_f16(d) >> 8);
            for (size_t j = 0; j < 16; ++j) {
                int n0 = 8, n1 = 8;
                const size_t i0 = b * 32 + j;
                const size_t i1 = b * 32 + j + 16;
                if (i0 < K) {
                    int qq = static_cast<int>(std::lrint(src[i0] / d));
                    qq = std::max(-8, std::min(7, qq));
                    n0 = qq + 8;
                }
                if (i1 < K) {
                    int qq = static_cast<int>(std::lrint(src[i1] / d));
                    qq = std::max(-8, std::min(7, qq));
                    n1 = qq + 8;
                }
                blk[2 + j] = static_cast<uint8_t>(n0 | (n1 << 4));
            }
        }
    }
    return q;
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
        bool lm_head_filled = false;
        if (safetensors_file.tensors().count("lm_head.weight") > 0) {
            copy_into_param(safetensors_file, "lm_head.weight", model->lm_head()->data.data(), cfg.vocab_size * cfg.hidden_size, true);
            lm_head_filled = true;
        } else if (safetensors_file.tensors().count("model.embed_tokens.weight") > 0) {
            copy_into_param(safetensors_file, "model.embed_tokens.weight", model->lm_head()->data.data(), cfg.vocab_size * cfg.hidden_size, true);
            lm_head_filled = true;
        }
        if (lm_head_filled) { build_lm_head_t(model); }
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

// ---------------------------------------------------------------------------
// GGUF loader: dequantise all tensors and map them into the LLaMAModel using
// the same safetensors-compatible naming convention (HF layout).
// ---------------------------------------------------------------------------
bool load_llama_weights_from_gguf_file(const std::string &gguf_path, model::LLaMAModel *model)
{
    if (model == nullptr) { return false; }

    std::cout << "[Loader] Detected GGUF file: " << gguf_path << "\n";
    std::unordered_map<std::string, quant::QuantWeight> qmap;
    auto tensors = io::load_all_tensors(gguf_path, &qmap);
    if (tensors.empty()) {
        std::cerr << "[Loader] GGUF: no tensors loaded from " << gguf_path << "\n";
        return false;
    }

    const model::LLamaConfig &cfg = model->config();
    const size_t vocab_size  = cfg.vocab_size;
    const size_t hidden      = cfg.hidden_size;
    const size_t num_layers  = cfg.num_hidden_layers;
    const size_t inter_size  = cfg.intermediate_size;
    const size_t head_dim    = hidden / cfg.num_attention_heads;
    const size_t kv_dim      = cfg.num_key_value_heads * head_dim;

    // Helper: copy or transpose a dequantised float tensor from the map into a raw buffer.
    auto copy_tensor = [&](const std::string &name, float *dst, size_t rows, size_t cols, bool should_transpose,
                           bool allow_prefix = false) -> bool {
        auto it = tensors.find(name);
        if (it == tensors.end()) { return false; }
        const core::Tensor &t = it->second;
        size_t expected = rows * cols;
        size_t numel = 1;
        for (size_t d : t.shape()) numel *= d;
        // Some GGUFs pad the vocabulary (e.g. Qwen 151665 -> 151936); the
        // padding sits at the end, so a larger tensor can be safely truncated.
        if (numel > expected && allow_prefix) { numel = expected; }
        if (numel != expected) {
            std::cerr << "[Loader] GGUF: shape mismatch for '" << name
                      << "' expected=" << expected << " got=" << numel << "\n";
            return false;
        }
        if (!should_transpose) {
            std::copy(t.data(), t.data() + numel, dst);
        } else {
            transpose_hf_to_cpp(t.data(), dst, rows, cols);
        }
        return true;
    };

    // Embedding (no transpose: rows=vocab_size, cols=hidden)
    if (model->embed_tokens() != nullptr) {
        copy_tensor("token_embd.weight", model->embed_tokens()->data.data(), vocab_size, hidden, false, true);
    }

    // Only the "llama" arch family stores Q/K with RoPE head_dim pairs
    // permuted in GGUF (verified against Qwen2 GGUFs, which are stored in
    // plain HF order). Other archs must NOT be unpermuted.
    const bool rope_perm = (io::gguf_architecture(gguf_path) == "llama");

    // Layers
    for (size_t l = 0; l < num_layers; ++l) {
        if (l >= model->layers().size()) break;
        model::LLaMABlock *block = model->layers()[l].get();
        std::string pfx = "blk." + std::to_string(l) + ".";

        copy_tensor(pfx + "attn_norm.weight",   block->input_layernorm_weight->data.data(),         1, hidden, false);
        copy_tensor(pfx + "attn_q.weight",       block->q_proj->data.data(),       hidden, hidden, true);
        copy_tensor(pfx + "attn_k.weight",       block->k_proj->data.data(),       kv_dim, hidden, true);
        // GGUF permutes Q/K head_dim pairs for RoPE (llama arch only); restore HF ordering.
        if (rope_perm) {
            unpermute_gguf_rope_heads(block->q_proj->data.data(), hidden, hidden, head_dim);
            unpermute_gguf_rope_heads(block->k_proj->data.data(), hidden, kv_dim, head_dim);
        }
        copy_tensor(pfx + "attn_v.weight",       block->v_proj->data.data(),       kv_dim, hidden, true);
        copy_tensor(pfx + "attn_q.bias",         block->q_bias->data.data(),       1, hidden, false);
        copy_tensor(pfx + "attn_k.bias",         block->k_bias->data.data(),       1, kv_dim, false);
        copy_tensor(pfx + "attn_v.bias",         block->v_bias->data.data(),       1, kv_dim, false);
        copy_tensor(pfx + "attn_output.weight",  block->o_proj->data.data(),       hidden, hidden, true);
        copy_tensor(pfx + "ffn_norm.weight",     block->post_attention_layernorm_weight->data.data(), 1, hidden, false);
        copy_tensor(pfx + "ffn_gate.weight",     block->gate_proj->data.data(),    inter_size, hidden, true);
        copy_tensor(pfx + "ffn_up.weight",       block->up_proj->data.data(),      inter_size, hidden, true);
        copy_tensor(pfx + "ffn_down.weight",     block->down_proj->data.data(),    hidden, inter_size, true);
    }

    // Final norm and lm_head
    if (model->norm_weight() != nullptr) {
        copy_tensor("output_norm.weight", model->norm_weight()->data.data(), 1, hidden, false);
    }
    if (model->lm_head() != nullptr) {
        if (!copy_tensor("output.weight", model->lm_head()->data.data(), vocab_size, hidden, true, true)) {
            // Weight-tying fallback: reuse embedding weights transposed
            copy_tensor("token_embd.weight", model->lm_head()->data.data(), vocab_size, hidden, true, true);
        }
        build_lm_head_t(model);
    }

    // Populate packed-quantized weights for the fast M==1 decode path.
    // GGUF stores Q/K rows with RoPE pairs interleaved; reorder rows to HF
    // layout (same permutation as unpermute_gguf_rope_heads on the FP32 path).
    auto take_quant = [&](const std::string &name, bool rope_perm, size_t exp_in, size_t exp_out) -> quant::QuantWeight {
        auto it = qmap.find(name);
        if (it == qmap.end()) { return {}; }
        if (it->second.in != exp_in || it->second.out != exp_out) { return {}; }
        quant::QuantWeight w = it->second;
        if (rope_perm && head_dim >= 2 && (head_dim % 2) == 0 && w.out % head_dim == 0) {
            const size_t half = head_dim / 2;
            const size_t nh = w.out / head_dim;
            const size_t rb = w.row_bytes();
            std::vector<uint8_t> perm(w.bytes.size());
            for (size_t h = 0; h < nh; ++h) {
                for (size_t s = 0; s < head_dim; ++s) {
                    // HF row s <- GGUF row: evens at s/2, odds at (s-1)/2 + half
                    const size_t src_local = ((s & 1) == 0) ? (s / 2) : ((s - 1) / 2 + half);
                    const size_t dst = (h * head_dim + s) * rb;
                    const size_t src = (h * head_dim + src_local) * rb;
                    std::memcpy(perm.data() + dst, w.bytes.data() + src, rb);
                }
            }
            w.bytes = std::move(perm);
        }
        return w;
    };

    for (size_t l = 0; l < model->layers().size(); ++l) {
        model::LLaMABlock *block = model->layers()[l].get();
        const std::string p = "blk." + std::to_string(l) + ".";
        block->q_w     = take_quant(p + "attn_q.weight",       rope_perm, hidden,     hidden);
        block->k_w     = take_quant(p + "attn_k.weight",       rope_perm, hidden,     kv_dim);
        block->v_w     = take_quant(p + "attn_v.weight",       false, hidden,     kv_dim);
        block->o_w     = take_quant(p + "attn_output.weight",  false, hidden,     hidden);
        block->gate_w  = take_quant(p + "ffn_gate.weight",     false, hidden,     inter_size);
        block->up_w    = take_quant(p + "ffn_up.weight",       false, hidden,     inter_size);
        block->down_w  = take_quant(p + "ffn_down.weight",     false, inter_size, hidden);

        // Fuse row-compatible quantized weights into single GEMV calls.
        auto fuse_rows = [](const quant::QuantWeight &a, const quant::QuantWeight &b, const quant::QuantWeight &c) -> quant::QuantWeight {
            if (!a.valid() || !b.valid() || !c.valid()) { return {}; }
            if (a.type != b.type || a.type != c.type || a.in != b.in || a.in != c.in ||
                a.row_bytes() != b.row_bytes() || a.row_bytes() != c.row_bytes()) { return {}; }
            quant::QuantWeight f;
            f.type = a.type; f.in = a.in;
            f.out = a.out + b.out + c.out;
            f.bytes.reserve(a.bytes.size() + b.bytes.size() + c.bytes.size());
            f.bytes.insert(f.bytes.end(), a.bytes.begin(), a.bytes.end());
            f.bytes.insert(f.bytes.end(), b.bytes.begin(), b.bytes.end());
            f.bytes.insert(f.bytes.end(), c.bytes.begin(), c.bytes.end());
            return f;
        };
        auto fuse_rows2 = [](const quant::QuantWeight &a, const quant::QuantWeight &b) -> quant::QuantWeight {
            if (!a.valid() || !b.valid()) { return {}; }
            if (a.type != b.type || a.in != b.in || a.row_bytes() != b.row_bytes()) { return {}; }
            quant::QuantWeight f;
            f.type = a.type; f.in = a.in;
            f.out = a.out + b.out;
            f.bytes.reserve(a.bytes.size() + b.bytes.size());
            f.bytes.insert(f.bytes.end(), a.bytes.begin(), a.bytes.end());
            f.bytes.insert(f.bytes.end(), b.bytes.begin(), b.bytes.end());
            return f;
        };
        block->qkv_w     = fuse_rows(block->q_w, block->k_w, block->v_w);
        block->gate_up_w = fuse_rows2(block->gate_w, block->up_w);
        // Mixed-type fusion fallback: Q4_K q/k can't fuse with a Q4_1 v
        // (Q6_K-derived). Expand Q4_K members to Q4_1 and retry — the fused
        // GEMV needs one kind per launch.
        if (!block->qkv_w.valid()) {
            auto as_q41 = [](const quant::QuantWeight &w) -> quant::QuantWeight {
                return (w.type == quant::QuantType::Q4_K)
                    ? quant_weight_q4k_to_q41(w) : w;
            };
            block->qkv_w = fuse_rows(as_q41(block->q_w), as_q41(block->k_w), as_q41(block->v_w));
        }
        if (!block->gate_up_w.valid()) {
            auto as_q41 = [](const quant::QuantWeight &w) -> quant::QuantWeight {
                return (w.type == quant::QuantType::Q4_K)
                    ? quant_weight_q4k_to_q41(w) : w;
            };
            block->gate_up_w = fuse_rows2(as_q41(block->gate_w), as_q41(block->up_w));
        }
    }
    if (model->lm_head() != nullptr) {
        quant::QuantWeight lq = take_quant("output.weight", false, hidden, vocab_size);
        if (!lq.valid()) { lq = take_quant("token_embd.weight", false, hidden, vocab_size); }
        // A Q8_0/Q6_K-derived token_embd reused as lm_head costs ~2x the
        // per-token DRAM traffic of Q4_0; requantize from the already-
        // dequantized FP32 head. Also fires when no packed weight exists at
        // all (e.g. Q6_K embd, which we dequantize but don't pack).
        if ((!lq.valid() || lq.type != quant::QuantType::Q4_0) &&
            model->lm_head_t().size() == vocab_size * hidden) {
            lq = quantize_fp32_to_q40(model->lm_head_t().data(), vocab_size, hidden);
        }
        if (lq.valid()) { model->lm_head_q() = std::move(lq); }
    }

    std::cout << "[Loader] GGUF: loaded " << tensors.size() << " dequantised tensors from " << gguf_path << "\n";
    return true;
}

bool load_llama_weights_from_dir(const std::string &dir_path, model::LLaMAModel *model)
{
    if (model == nullptr) { return false; }
    bool any_file_loaded = false;
    size_t files_attempted = 0;
    size_t files_loaded = 0;

    try {
        // --- Priority 1: load a single GGUF file (quantised, faster) ---
        for (const auto &entry : fs::directory_iterator(dir_path)) {
            if (!entry.is_regular_file()) { continue; }
            std::string path_str = entry.path().string();
            if (path_str.size() < 5 || path_str.compare(path_str.size() - 5, 5, ".gguf") != 0) { continue; }

            files_attempted++;
            std::cout << "[Loader] Attempting to load GGUF: " << path_str << "\n";
            if (load_llama_weights_from_gguf_file(path_str, model)) {
                any_file_loaded = true;
                files_loaded++;
                std::cout << "[Loader] Successfully loaded GGUF: " << path_str << "\n";
            } else {
                std::cerr << "[Loader] Failed to load GGUF file: " << path_str << "\n";
            }
        }

        // --- Priority 2: fallback to safetensors shards ---
        if (!any_file_loaded) {
            for (const auto &entry : fs::directory_iterator(dir_path)) {
                if (!entry.is_regular_file()) { continue; }
                std::string path_str = entry.path().string();
                if (path_str.size() < 12 || path_str.compare(path_str.size() - 12, 12, ".safetensors") != 0) { continue; }

                files_attempted++;
                std::cout << "[Loader] Attempting to load: " << path_str << "\n";

                SafetensorsFile safetensors_file;
                if (load_safetensors(path_str, &safetensors_file)) {
                    any_file_loaded = true;
                    files_loaded++;
                    load_one_file_into_model(safetensors_file, model);
                    std::cout << "[Loader] Successfully loaded: " << path_str << "\n";
                } else {
                    std::cerr << "[Loader] Failed to load safetensors file: " << path_str << "\n";
                }
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "[Loader] Error loading weights from directory: " << e.what() << "\n";
        return false;
    }

    std::cout << "[Loader] Weight loading summary: " << files_loaded << "/" << files_attempted << " files loaded\n";
    return any_file_loaded;
}

}  // namespace io
}  // namespace serius
