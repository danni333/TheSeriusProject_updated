#include "gguf_export.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf_loader.h"
#include "safetensors_writer.h"

namespace fs = std::filesystem;

namespace serius {
namespace io {

namespace {

/// GGUF dequantized tensors are stored [out][in] row-major — identical to
/// the HF safetensors layout — so 2-D exports only need the dims swapped in
/// the shape metadata and (for llama arch) a row permutation on Q/K.
struct ExportEntry
{
    std::string hf_name;
    core::Tensor tensor;  // FP32, data already in [out][in] order
};

/// Inverse of the GGUF RoPE head-dim interleave, applied to [out][in] data:
/// within each head, output row s is taken from GGUF row src(s). Works on a
/// per-head temporary block so the permutation cannot corrupt rows already
/// written.
void unpermute_qk_rows(float *data, size_t out_dim, size_t in_dim, size_t head_dim)
{
    if (head_dim < 2 || (head_dim % 2) != 0 || out_dim % head_dim != 0) {
        return;
    }
    const size_t half = head_dim / 2;
    const size_t n_heads = out_dim / head_dim;
    std::vector<float> block(head_dim * in_dim);
    for (size_t h = 0; h < n_heads; ++h) {
        float *base = data + h * head_dim * in_dim;
        std::memcpy(block.data(), base, block.size() * sizeof(float));
        for (size_t s = 0; s < head_dim; ++s) {
            const size_t src_local = ((s & 1) == 0) ? (s / 2) : ((s - 1) / 2 + half);
            std::memcpy(base + s * in_dim, block.data() + src_local * in_dim, in_dim * sizeof(float));
        }
    }
}

std::string map_tensor_name(const std::string &gguf_name, int num_layers)
{
    static const std::unordered_map<std::string, std::string> fixed = {
        {"token_embd.weight", "model.embed_tokens.weight"},
        {"output_norm.weight", "model.norm.weight"},
        {"output.weight", "lm_head.weight"},
    };
    auto it = fixed.find(gguf_name);
    if (it != fixed.end()) {
        return it->second;
    }

    // blk.N.<rest>
    if (gguf_name.rfind("blk.", 0) != 0) {
        return {};
    }
    const size_t dot1 = gguf_name.find('.', 4);
    if (dot1 == std::string::npos) {
        return {};
    }
    int layer;
    try {
        layer = std::stoi(gguf_name.substr(4, dot1 - 4));
    }
    catch (...) {
        return {};
    }
    if (layer < 0 || layer >= num_layers) {
        return {};
    }
    const std::string rest = gguf_name.substr(dot1 + 1);
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";

    static const std::unordered_map<std::string, std::string> per_layer = {
        {"attn_norm.weight", "input_layernorm.weight"},
        {"attn_q.weight", "self_attn.q_proj.weight"},
        {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_v.weight", "self_attn.v_proj.weight"},
        {"attn_q.bias", "self_attn.q_proj.bias"},
        {"attn_k.bias", "self_attn.k_proj.bias"},
        {"attn_v.bias", "self_attn.v_proj.bias"},
        {"attn_output.weight", "self_attn.o_proj.weight"},
        {"ffn_norm.weight", "post_attention_layernorm.weight"},
        {"ffn_gate.weight", "mlp.gate_proj.weight"},
        {"ffn_up.weight", "mlp.up_proj.weight"},
        {"ffn_down.weight", "mlp.down_proj.weight"},
    };
    auto jt = per_layer.find(rest);
    if (jt == per_layer.end()) {
        return {};
    }
    return prefix + jt->second;
}

void write_config_json(const std::string &path, const model::ModelConfig &cfg)
{
    nlohmann::json j;
    j["vocab_size"] = cfg.vocab_size;
    j["hidden_size"] = cfg.d_model;
    j["num_attention_heads"] = cfg.n_heads;
    j["num_key_value_heads"] = cfg.n_kv_heads;
    j["num_hidden_layers"] = cfg.n_layers;
    j["intermediate_size"] = cfg.d_ff;
    j["max_position_embeddings"] = cfg.max_seq_len;
    j["rms_norm_eps"] = cfg.rms_norm_eps;
    j["rope_theta"] = cfg.rope_theta;
    j["architectures"] = {"LlamaForCausalLM"};
    j["model_type"] = "llama";

    std::ofstream out(path);
    out << j.dump(2);
}

void copy_if_exists(const fs::path &src, const fs::path &dst_dir)
{
    std::error_code ec;
    if (fs::exists(src, ec) && fs::is_regular_file(src, ec)) {
        fs::copy_file(src, dst_dir / src.filename(), fs::copy_options::overwrite_existing, ec);
    }
}

}  // namespace

bool export_gguf_to_hf_dir(const std::string &gguf_path, const std::string &out_dir)
{
    GGUFMeta meta = load_gguf_metadata(gguf_path);
    if (!meta.valid) {
        std::cerr << "[convert] not a valid GGUF file: " << gguf_path << "\n";
        return false;
    }

    const std::string arch = gguf_architecture(gguf_path);
    const model::ModelConfig cfg = gguf_to_model_config(gguf_path);
    if (cfg.n_layers == 0 || cfg.d_model == 0) {
        std::cerr << "[convert] could not read model config from GGUF metadata\n";
        return false;
    }

    std::unordered_map<std::string, core::Tensor> tensors;
    try {
        tensors = load_all_tensors(gguf_path, nullptr);
    }
    catch (const std::exception &e) {
        std::cerr << "[convert] GGUF load failed: " << e.what() << "\n";
        return false;
    }
    if (tensors.empty()) {
        std::cerr << "[convert] no tensors loaded from " << gguf_path << "\n";
        return false;
    }

    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "[convert] cannot create output dir: " << out_dir << "\n";
        return false;
    }

    const bool rope_perm = (arch == "llama");
    const size_t head_dim = cfg.d_model / cfg.n_heads;

    std::vector<std::pair<std::string, core::Tensor>> out_tensors;
    out_tensors.reserve(tensors.size());
    size_t skipped = 0;

    for (auto &[gguf_name, tensor] : tensors) {
        const std::string hf_name = map_tensor_name(gguf_name, static_cast<int>(cfg.n_layers));
        if (hf_name.empty()) {
            ++skipped;
            continue;
        }

        // GGUF shape is {in, out}; HF safetensors expects {out, in}.
        const std::vector<size_t> shape = tensor.shape();
        const std::vector<size_t> hf_shape = (shape.size() == 2) ? std::vector<size_t>{shape[1], shape[0]} : shape;
        core::Tensor exported(hf_shape, hal::DType::Float32);
        const size_t numel = tensor.size();
        std::memcpy(exported.data(), tensor.data(), numel * sizeof(float));

        // Undo the GGUF RoPE head-dim interleave on Q/K weights (llama arch).
        // Biases are stored unpermuted in GGUF and copied as-is.
        if (rope_perm && shape.size() == 2) {
            const bool is_q = (gguf_name.find("attn_q.weight") != std::string::npos);
            const bool is_k = (gguf_name.find("attn_k.weight") != std::string::npos);
            if (is_q || is_k) {
                const size_t out_dim = shape[1];
                const size_t in_dim = shape[0];
                const size_t expected_heads = is_q ? cfg.n_heads : cfg.n_kv_heads;
                if (out_dim == expected_heads * head_dim) {
                    unpermute_qk_rows(exported.data(), out_dim, in_dim, head_dim);
                }
            }
        }

        out_tensors.emplace_back(hf_name, std::move(exported));
    }

    if (out_tensors.empty()) {
        std::cerr << "[convert] no mappable tensors found in " << gguf_path << "\n";
        return false;
    }

    std::sort(out_tensors.begin(), out_tensors.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

    const std::string st_path = (fs::path(out_dir) / "model.safetensors").string();
    SafetensorsWriter writer;
    for (const auto &[name, tensor] : out_tensors) {
        writer.add_tensor(name, tensor);
    }
    writer.save(st_path);

    write_config_json((fs::path(out_dir) / "config.json").string(), cfg);

    // Tokenizer files live next to the GGUF in typical model dirs.
    const fs::path gguf_dir = fs::path(gguf_path).parent_path();
    const fs::path out_path = fs::path(out_dir);
    copy_if_exists(gguf_dir / "tokenizer.json", out_path);
    copy_if_exists(gguf_dir / "tokenizer_config.json", out_path);
    copy_if_exists(gguf_dir / "special_tokens_map.json", out_path);
    copy_if_exists(gguf_dir / "vocab.json", out_path);
    copy_if_exists(gguf_dir / "merges.txt", out_path);

    std::cout << "[convert] exported " << out_tensors.size() << " tensors"
              << (skipped ? " (" + std::to_string(skipped) + " skipped)" : "") << " to " << out_dir << "\n";
    return true;
}

}  // namespace io
}  // namespace serius
