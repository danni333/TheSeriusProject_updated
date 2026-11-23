#include "gguf_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../tensor/tensor.h"
#include "gguf_loader.h"
#include "safetensors.h"
#include "safetensors_loader.h"
#include "sharding.h"

namespace fs = std::filesystem;

namespace serius {
namespace io {

namespace {

// GGUF value types (ggml spec numbering — matches what gguf_loader reads).
constexpr uint32_t kValU8 = 0, kValI8 = 1, kValU16 = 2, kValI16 = 3;
constexpr uint32_t kValU32 = 4, kValI32 = 5, kValF32 = 6, kValBool = 7;
constexpr uint32_t kValStr = 8, kValArr = 9, kValU64 = 10, kValI64 = 11, kValF64 = 12;

constexpr uint64_t kAlignment = 32;

void put_u16(std::string &o, uint16_t v)
{
    o.append(reinterpret_cast<const char *>(&v), 2);
}
void put_u32(std::string &o, uint32_t v)
{
    o.append(reinterpret_cast<const char *>(&v), 4);
}
void put_u64(std::string &o, uint64_t v)
{
    o.append(reinterpret_cast<const char *>(&v), 8);
}
void put_i32(std::string &o, int32_t v)
{
    o.append(reinterpret_cast<const char *>(&v), 4);
}
void put_f32(std::string &o, float v)
{
    o.append(reinterpret_cast<const char *>(&v), 4);
}

void put_str(std::string &o, const std::string &s)
{
    put_u64(o, s.size());
    o.append(s);
}

void kv_begin(std::string &o, const std::string &key, uint32_t type)
{
    put_str(o, key);
    put_u32(o, type);
}

void kv_u32(std::string &o, const std::string &key, uint32_t v)
{
    kv_begin(o, key, kValU32);
    put_u32(o, v);
}

void kv_i32(std::string &o, const std::string &key, int32_t v)
{
    kv_begin(o, key, kValI32);
    put_i32(o, v);
}

void kv_f32(std::string &o, const std::string &key, float v)
{
    kv_begin(o, key, kValF32);
    put_f32(o, v);
}

void kv_str(std::string &o, const std::string &key, const std::string &v)
{
    kv_begin(o, key, kValStr);
    put_str(o, v);
}

void kv_str_array(std::string &o, const std::string &key, const std::vector<std::string> &vals)
{
    kv_begin(o, key, kValArr);
    put_u32(o, kValStr);
    put_u64(o, vals.size());
    for (const auto &v : vals) {
        put_str(o, v);
    }
}

void kv_i32_array(std::string &o, const std::string &key, const std::vector<int32_t> &vals)
{
    kv_begin(o, key, kValArr);
    put_u32(o, kValI32);
    put_u64(o, vals.size());
    for (int32_t v : vals) {
        put_i32(o, v);
    }
}

uint64_t align_up(uint64_t v, uint64_t a)
{
    return (v + a - 1) / a * a;
}

/// IEEE-754 binary32 -> binary16, round-to-nearest-even.
uint16_t f32_to_f16_rn(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = static_cast<int>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        // subnormal or zero: mant/2^23 * 2^(exp-15) -> shift into 10-bit field
        const uint32_t m = mant | 0x800000u;
        const int shift = 14 - exp;  // bits to drop from the 24-bit mantissa
        uint32_t half_mant = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_mant & 1u))) {
            ++half_mant;
        }
        return static_cast<uint16_t>(sign | half_mant);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    // round mantissa 23 -> 10 bits
    uint32_t hm = mant >> 13;
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
        ++hm;
        if (hm == 0x400u) {  // mantissa overflow -> bump exponent
            hm = 0;
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp + 1) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | hm);
}

/// Inverse of gguf_export's unpermute_qk_rows: HF [out][in] row-major ->
/// GGUF RoPE-interleaved row order (used by the "llama" arch family).
/// For each head, GGUF row src(s) receives HF row s, where
/// src(s) = s/2 (even s) or (s-1)/2 + half (odd s).
void permute_qk_rows_gguf(float *data, size_t out_dim, size_t in_dim, size_t head_dim)
{
    if (head_dim < 2 || (head_dim % 2) != 0 || out_dim % head_dim != 0) {
        return;
    }
    const size_t half = head_dim / 2;
    const size_t n_heads = out_dim / head_dim;
    std::vector<float> src_rows(head_dim * in_dim);
    for (size_t h = 0; h < n_heads; ++h) {
        float *base = data + h * head_dim * in_dim;
        std::memcpy(src_rows.data(), base, src_rows.size() * sizeof(float));
        for (size_t s = 0; s < head_dim; ++s) {
            const size_t dst_local = ((s & 1) == 0) ? (s / 2) : ((s - 1) / 2 + half);
            std::memcpy(base + dst_local * in_dim, src_rows.data() + s * in_dim, in_dim * sizeof(float));
        }
    }
}

/// Map a HuggingFace tensor name to the GGUF name our loader expects.
/// Inverse of map_tensor_name() in gguf_export.cpp.
std::string hf_to_gguf_name(const std::string &hf_name, int num_layers)
{
    static const std::unordered_map<std::string, std::string> fixed = {
        {"model.embed_tokens.weight", "token_embd.weight"},
        {"model.norm.weight", "output_norm.weight"},
        {"lm_head.weight", "output.weight"},
    };
    auto it = fixed.find(hf_name);
    if (it != fixed.end()) {
        return it->second;
    }

    const std::string prefix = "model.layers.";
    if (hf_name.rfind(prefix, 0) != 0) {
        return {};
    }
    const size_t dot = hf_name.find('.', prefix.size());
    if (dot == std::string::npos) {
        return {};
    }
    int layer;
    try {
        layer = std::stoi(hf_name.substr(prefix.size(), dot - prefix.size()));
    }
    catch (...) {
        return {};
    }
    if (layer < 0 || layer >= num_layers) {
        return {};
    }
    const std::string rest = hf_name.substr(dot + 1);
    const std::string gprefix = "blk." + std::to_string(layer) + ".";

    static const std::unordered_map<std::string, std::string> per_layer = {
        {"input_layernorm.weight", "attn_norm.weight"},
        {"self_attn.q_proj.weight", "attn_q.weight"},
        {"self_attn.k_proj.weight", "attn_k.weight"},
        {"self_attn.v_proj.weight", "attn_v.weight"},
        {"self_attn.q_proj.bias", "attn_q.bias"},
        {"self_attn.k_proj.bias", "attn_k.bias"},
        {"self_attn.v_proj.bias", "attn_v.bias"},
        {"self_attn.o_proj.weight", "attn_output.weight"},
        {"post_attention_layernorm.weight", "ffn_norm.weight"},
        {"mlp.gate_proj.weight", "ffn_gate.weight"},
        {"mlp.up_proj.weight", "ffn_up.weight"},
        {"mlp.down_proj.weight", "ffn_down.weight"},
    };
    auto jt = per_layer.find(rest);
    if (jt == per_layer.end()) {
        return {};
    }
    return gprefix + jt->second;
}

struct ModelDims
{
    size_t vocab_size = 0;
    size_t hidden = 0;
    size_t layers = 0;
    size_t heads = 0;
    size_t kv_heads = 0;
    size_t inter = 0;
    size_t max_pos = 0;
    float rms_eps = 1e-5f;
    float rope_theta = 10000.0f;
    std::string arch;
};

std::string derive_arch(const nlohmann::json &cfg)
{
    std::string mt = cfg.value("model_type", "");
    if (!mt.empty()) {
        return mt;
    }
    if (cfg.contains("architectures") && cfg["architectures"].is_array() && !cfg["architectures"].empty()) {
        std::string a = cfg["architectures"][0].get<std::string>();
        std::transform(a.begin(), a.end(), a.begin(), ::tolower);
        const std::string suf = "forcausallm";
        if (a.size() > suf.size() && a.compare(a.size() - suf.size(), suf.size(), suf) == 0) {
            a.resize(a.size() - suf.size());
        }
        return a;
    }
    return {};
}

bool load_config(const fs::path &dir, ModelDims *dims)
{
    const fs::path cfg_path = dir / "config.json";
    std::ifstream in(cfg_path);
    if (!in) {
        std::cerr << "[GGUFWriter] missing config.json in " << dir << "\n";
        return false;
    }
    nlohmann::json j;
    try {
        in >> j;
    }
    catch (const std::exception &e) {
        std::cerr << "[GGUFWriter] cannot parse " << cfg_path << ": " << e.what() << "\n";
        return false;
    }
    dims->vocab_size = j.value("vocab_size", 0);
    dims->hidden = j.value("hidden_size", j.value("n_embd", 0));
    dims->layers = j.value("num_hidden_layers", j.value("n_layer", 0));
    dims->heads = j.value("num_attention_heads", j.value("n_head", 0));
    dims->kv_heads = j.value("num_key_value_heads", dims->heads);
    dims->inter = j.value("intermediate_size", j.value("n_ff", 0));
    dims->max_pos = j.value("max_position_embeddings", 0);
    dims->rms_eps = j.value("rms_norm_eps", dims->rms_eps);
    dims->rope_theta = j.value("rope_theta", dims->rope_theta);
    dims->arch = derive_arch(j);
    return dims->hidden > 0 && dims->layers > 0;
}

/// Load all safetensors weights of a HF model dir as FP32 tensors keyed by
/// HF name. Supports a single .safetensors file or a shard index manifest.
bool load_hf_tensors(const fs::path &dir, const fs::path &explicit_file, std::map<std::string, core::Tensor> *out)
{
    out->clear();

    // Sharded layout: *.safetensors.index.json
    for (const auto &entry : fs::directory_iterator(dir)) {
        const std::string fname = entry.path().filename().string();
        if (fname.size() > 19 && fname.compare(fname.size() - 19, 19, ".safetensors.index.json") == 0) {
            if (!load_sharded_safetensors(entry.path().string(), out)) {
                std::cerr << "[GGUFWriter] failed to load shard index " << entry.path() << "\n";
                return false;
            }
            return !out->empty();
        }
    }

    // Single file: explicit path or model.safetensors / first *.safetensors
    fs::path st_path = explicit_file;
    if (st_path.empty()) {
        st_path = dir / "model.safetensors";
        if (!fs::exists(st_path)) {
            for (const auto &entry : fs::directory_iterator(dir)) {
                const std::string fname = entry.path().filename().string();
                if (fname.size() > 12 && fname.compare(fname.size() - 12, 12, ".safetensors") == 0) {
                    st_path = entry.path();
                    break;
                }
            }
        }
    }
    if (st_path.empty() || !fs::exists(st_path)) {
        std::cerr << "[GGUFWriter] no .safetensors file found in " << dir << "\n";
        return false;
    }

    SafetensorsFile file;
    if (!load_safetensors(st_path.string(), &file)) {
        std::cerr << "[GGUFWriter] cannot load " << st_path << "\n";
        return false;
    }
    for (const auto &[name, meta] : file.tensors()) {
        size_t numel = 1;
        std::vector<size_t> shape;
        for (int64_t d : meta.shape) {
            if (d <= 0) {
                numel = 0;
                break;
            }
            shape.push_back(static_cast<size_t>(d));
            numel *= shape.back();
        }
        if (numel == 0) {
            continue;
        }
        core::Tensor t(shape, hal::DType::Float32);
        if (!file.copy_tensor_to_float(name, t.data(), numel)) {
            continue;
        }
        out->emplace(name, std::move(t));
    }
    return !out->empty();
}

struct TokenizerInfo
{
    std::vector<std::string> tokens;  // index = token id
    std::vector<int32_t> token_types;
    std::vector<std::string> merges;
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    bool valid = false;
};

std::string json_token_string(const nlohmann::json &v)
{
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_number_integer()) {
        return std::to_string(v.get<int64_t>());
    }
    if (v.is_object() && v.contains("content")) {
        return v["content"].get<std::string>();
    }
    if (v.is_array() && !v.empty()) {
        return json_token_string(v[0]);
    }
    return {};
}

bool looks_special(const std::string &tok)
{
    return tok == "<s>" || tok == "</s>" || tok == "<unk>" || tok == "<pad>" ||
           (tok.size() > 2 && tok.front() == '<' && tok.back() == '>' && tok.find("|") != std::string::npos);
}

TokenizerInfo load_tokenizer_info(const fs::path &dir)
{
    TokenizerInfo info;

    std::unordered_map<std::string, int32_t> vocab;
    std::vector<std::pair<std::string, int32_t>> added;
    std::set<std::string> added_set;

    // tokenizer.json — HF fast tokenizer or our internal format
    const fs::path tok_path = dir / "tokenizer.json";
    if (fs::exists(tok_path)) {
        std::ifstream in(tok_path);
        nlohmann::json j;
        try {
            in >> j;
        }
        catch (...) {
            j = nlohmann::json();
        }

        if (j.contains("model") && j["model"].is_object() && j["model"].contains("vocab")) {
            for (auto it = j["model"]["vocab"].begin(); it != j["model"]["vocab"].end(); ++it) {
                vocab[it.key()] = it.value().get<int32_t>();
            }
            if (j["model"].contains("merges")) {
                for (const auto &m : j["model"]["merges"]) {
                    if (m.is_string()) {
                        info.merges.push_back(m.get<std::string>());
                    }
                }
            }
        } else if (j.contains("vocab") && j["vocab"].is_object()) {
            for (auto it = j["vocab"].begin(); it != j["vocab"].end(); ++it) {
                vocab[it.key()] = it.value().get<int32_t>();
            }
        } else if (j.contains("vocab") && j["vocab"].is_array()) {
            // internal save() format: [{"t": bytes, "i": id}]
            for (const auto &item : j["vocab"]) {
                if (!item.contains("t") || !item.contains("i")) {
                    continue;
                }
                std::vector<uint8_t> bytes = item["t"].get<std::vector<uint8_t>>();
                vocab[std::string(bytes.begin(), bytes.end())] = item["i"].get<int32_t>();
            }
            if (j.contains("merges") && j["merges"].is_array()) {
                // merges stored as [id_a, id_b, id_c] triples
                std::unordered_map<int32_t, std::string> id_tok;
                for (const auto &[tok, id] : vocab) {
                    id_tok[id] = tok;
                }
                for (const auto &m : j["merges"]) {
                    if (m.is_array() && m.size() == 3 && m[0].is_number() && m[1].is_number() && m[2].is_number()) {
                        const int32_t a = m[0].get<int32_t>();
                        const int32_t b = m[1].get<int32_t>();
                        if (id_tok.count(a) && id_tok.count(b)) {
                            info.merges.push_back(id_tok[a] + " " + id_tok[b]);
                        }
                    }
                }
            }
        }
        if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
            for (const auto &item : j["added_tokens"]) {
                if (!item.contains("content")) {
                    continue;
                }
                const std::string content = item["content"].get<std::string>();
                int32_t id = -1;
                if (item.contains("id") && item["id"].is_number()) {
                    id = item["id"].get<int32_t>();
                }
                vocab[content] = id >= 0 ? id : static_cast<int32_t>(vocab.size());
                added.emplace_back(content, vocab[content]);
                added_set.insert(content);
            }
        }
    }

    // merges.txt fallback (LLaMA/SentencePiece layout): "a b" per line
    if (info.merges.empty()) {
        const fs::path merges_path = dir / "merges.txt";
        if (fs::exists(merges_path)) {
            std::ifstream in(merges_path);
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                info.merges.push_back(line);
            }
        }
    }

    if (vocab.empty()) {
        return info;
    }

    size_t max_id = 0;
    for (const auto &[tok, id] : vocab) {
        if (id >= 0 && static_cast<size_t>(id) > max_id) {
            max_id = static_cast<size_t>(id);
        }
    }
    // contiguous ids only
    std::vector<std::string> toks(max_id + 1);
    std::vector<char> seen(max_id + 1, 0);
    for (const auto &[tok, id] : vocab) {
        if (id >= 0 && static_cast<size_t>(id) <= max_id) {
            toks[id] = tok;
            seen[id] = 1;
        }
    }
    for (size_t i = 0; i < toks.size(); ++i) {
        if (!seen[i]) {
            toks[i] = "";
        }
    }
    info.tokens = std::move(toks);
    info.token_types.assign(info.tokens.size(), 1);
    for (const auto &[tok, id] : added) {
        if (id >= 0 && static_cast<size_t>(id) < info.token_types.size()) {
            info.token_types[id] = 3;
        }
    }
    for (size_t i = 0; i < info.tokens.size(); ++i) {
        if (info.token_types[i] == 1 && looks_special(info.tokens[i])) {
            info.token_types[i] = 3;
        }
    }

    // bos/eos from tokenizer_config.json
    const fs::path tcfg_path = dir / "tokenizer_config.json";
    if (fs::exists(tcfg_path)) {
        std::ifstream in(tcfg_path);
        nlohmann::json j;
        try {
            in >> j;
        }
        catch (...) {
            j = nlohmann::json();
        }
        auto resolve = [&](const char *key) -> int32_t {
            if (!j.contains(key)) {
                return -1;
            }
            const std::string s = json_token_string(j[key]);
            if (s.empty()) {
                return -1;
            }
            auto it = vocab.find(s);
            return it == vocab.end() ? -1 : it->second;
        };
        info.bos_id = resolve("bos_token");
        info.eos_id = resolve("eos_token");
    }

    info.valid = !info.tokens.empty();
    return info;
}

void append_f16(std::string &o, const float *src, size_t n)
{
    o.reserve(o.size() + n * 2);
    for (size_t i = 0; i < n; ++i) {
        const uint16_t h = f32_to_f16_rn(src[i]);
        put_u16(o, h);
    }
}

}  // namespace

bool write_hf_dir_to_gguf(const std::string &hf_dir, const std::string &out_path, const GGUFWritOptions &opt)
{
    fs::path in_path(hf_dir);
    fs::path dir = in_path;
    fs::path explicit_st;
    std::error_code ec;
    if (fs::is_regular_file(in_path, ec)) {
        dir = in_path.parent_path();
        if (in_path.extension() == ".safetensors") {
            explicit_st = in_path;
        }
    }
    if (!fs::is_directory(dir, ec)) {
        std::cerr << "[GGUFWriter] not a directory: " << dir << "\n";
        return false;
    }

    ModelDims dims;
    if (!load_config(dir, &dims)) {
        return false;
    }
    if (!opt.arch.empty()) {
        dims.arch = opt.arch;
    }
    if (dims.arch.empty()) {
        dims.arch = "llama";
        std::cerr << "[GGUFWriter] warning: no architecture in config.json, assuming llama\n";
    }

    std::map<std::string, core::Tensor> tensors;
    if (!load_hf_tensors(dir, explicit_st, &tensors)) {
        return false;
    }

    TokenizerInfo tok;
    if (opt.embed_tokenizer) {
        tok = load_tokenizer_info(dir);
        if (tok.valid && dims.vocab_size == 0) {
            dims.vocab_size = tok.tokens.size();
        }
    }

    // Select and rename tensors to GGUF names.
    struct OutTensor
    {
        std::string name;
        const core::Tensor *src;        // HF layout [out][in] row-major, FP32
        std::vector<size_t> gguf_dims;  // reversed HF shape
        bool permute_rope = false;
    };
    std::vector<OutTensor> outs;
    outs.reserve(tensors.size());
    size_t skipped = 0;
    const bool rope_perm = (dims.arch == "llama");
    const size_t head_dim = dims.heads ? dims.hidden / dims.heads : 0;

    for (const auto &[hf_name, tensor] : tensors) {
        const std::string gname = hf_to_gguf_name(hf_name, static_cast<int>(dims.layers));
        if (gname.empty()) {
            ++skipped;
            continue;
        }
        OutTensor ot;
        ot.name = gname;
        ot.src = &tensor;
        const auto &shape = tensor.shape();
        ot.gguf_dims.assign(shape.rbegin(), shape.rend());
        if (rope_perm && shape.size() == 2 &&
            (gname.find("attn_q.weight") != std::string::npos || gname.find("attn_k.weight") != std::string::npos)) {
            ot.permute_rope = true;
        }
        outs.push_back(std::move(ot));
    }
    if (outs.empty()) {
        std::cerr << "[GGUFWriter] no mappable tensors found\n";
        return false;
    }
    std::sort(outs.begin(), outs.end(), [](const OutTensor &a, const OutTensor &b) { return a.name < b.name; });

    // ---- KV section ----
    std::string kv;
    kv.reserve(4096);
    kv_str(kv, "general.architecture", dims.arch);
    kv_u32(kv, "general.file_type", opt.use_f32 ? 0u : 1u);

    const std::string p = dims.arch + ".";
    kv_u32(kv, p + "context_length", static_cast<uint32_t>(dims.max_pos));
    kv_u32(kv, p + "embedding_length", static_cast<uint32_t>(dims.hidden));
    kv_u32(kv, p + "block_count", static_cast<uint32_t>(dims.layers));
    kv_u32(kv, p + "feed_forward_length", static_cast<uint32_t>(dims.inter));
    if (dims.heads) {
        kv_u32(kv, p + "attention.head_count", static_cast<uint32_t>(dims.heads));
    }
    if (dims.kv_heads) {
        kv_u32(kv, p + "attention.head_count_kv", static_cast<uint32_t>(dims.kv_heads));
    }
    kv_f32(kv, p + "attention.layer_norm_rms_epsilon", dims.rms_eps);
    kv_f32(kv, p + "rope.freq_base", dims.rope_theta);

    if (tok.valid) {
        kv_str(kv, "tokenizer.ggml.model", "gpt2");
        kv_str_array(kv, "tokenizer.ggml.tokens", tok.tokens);
        kv_i32_array(kv, "tokenizer.ggml.token_type", tok.token_types);
        kv_str_array(kv, "tokenizer.ggml.merges", tok.merges);
        if (tok.bos_id >= 0) {
            kv_i32(kv, "tokenizer.ggml.bos_token_id", tok.bos_id);
        }
        if (tok.eos_id >= 0) {
            kv_i32(kv, "tokenizer.ggml.eos_token_id", tok.eos_id);
        }
    }

    // general.architecture + general.file_type + 6 unconditional arch keys
    // + conditional head keys + tokenizer entries.
    uint64_t n_kv = 2 + 6 + (dims.heads ? 1 : 0) + (dims.kv_heads ? 1 : 0);
    if (tok.valid) {
        n_kv += 4 + (tok.bos_id >= 0 ? 1 : 0) + (tok.eos_id >= 0 ? 1 : 0);
    }

    // ---- tensor infos: sizes first (for offsets), serialize once ----
    struct InfoRec
    {
        const OutTensor *ot;
        uint64_t offset = 0;
        uint64_t payload = 0;
        uint32_t ggml_type = 0;
    };
    std::vector<InfoRec> infos(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) {
        const OutTensor &ot = outs[i];
        uint64_t numel = 1;
        for (size_t d : ot.gguf_dims) {
            numel *= d;
        }
        infos[i].ot = &ot;
        infos[i].payload = opt.use_f32 ? numel * 4 : numel * 2;
        infos[i].ggml_type = opt.use_f32 ? static_cast<uint32_t>(GGMLType::F32) : static_cast<uint32_t>(GGMLType::F16);
    }

    const uint64_t header_size = 24;  // magic + version + n_tensors + n_kv
    // Tensor info record size is fixed per tensor once offsets are known;
    // iterate to a fixed point (offsets don't affect record size).
    std::string infos_buf;
    uint64_t data_base = 0;
    for (int pass = 0; pass < 2; ++pass) {
        infos_buf.clear();
        uint64_t off = 0;
        for (auto &rec : infos) {
            off = align_up(off, kAlignment);
            rec.offset = off;
            off += rec.payload;

            put_str(infos_buf, rec.ot->name);
            put_u32(infos_buf, static_cast<uint32_t>(rec.ot->gguf_dims.size()));
            for (size_t d : rec.ot->gguf_dims) {
                put_u64(infos_buf, d);
            }
            put_u32(infos_buf, rec.ggml_type);
            put_u64(infos_buf, rec.offset);
        }
        data_base = align_up(header_size + kv.size() + infos_buf.size(), kAlignment);
    }

    // ---- write file ----
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[GGUFWriter] cannot open " << out_path << " for writing\n";
        return false;
    }

    const uint32_t magic = 0x46554747;  // "GGUF"
    const uint32_t version = 3;
    const uint64_t n_tensors = infos.size();
    out.write(reinterpret_cast<const char *>(&magic), 4);
    out.write(reinterpret_cast<const char *>(&version), 4);
    out.write(reinterpret_cast<const char *>(&n_tensors), 8);
    out.write(reinterpret_cast<const char *>(&n_kv), 8);

    out.write(kv.data(), static_cast<std::streamsize>(kv.size()));
    out.write(infos_buf.data(), static_cast<std::streamsize>(infos_buf.size()));

    // pad to data_base
    const uint64_t written = header_size + kv.size() + infos_buf.size();
    const std::streampos pad_pos = static_cast<std::streampos>(data_base > written ? data_base - written : 0);
    if (pad_pos > 0) {
        const std::string pad(static_cast<size_t>(pad_pos), '\0');
        out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    }

    // tensor payloads
    std::string payload;
    for (const auto &rec : infos) {
        const OutTensor &ot = *rec.ot;
        const uint64_t cur = static_cast<uint64_t>(out.tellp());
        const uint64_t expected = data_base + rec.offset;
        if (cur < expected) {
            const std::string pad(static_cast<size_t>(expected - cur), '\0');
            out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
        }

        payload.clear();
        const size_t numel = ot.src->size();
        std::vector<float> buf;
        const float *src = ot.src->data();
        if (ot.permute_rope) {
            buf.assign(src, src + numel);
            const auto &shape = ot.src->shape();
            const size_t out_dim = shape[0];
            const size_t in_dim = shape.size() > 1 ? shape[1] : 1;
            permute_qk_rows_gguf(buf.data(), out_dim, in_dim, head_dim);
            src = buf.data();
        }
        if (opt.use_f32) {
            payload.append(reinterpret_cast<const char *>(src), numel * sizeof(float));
        } else {
            append_f16(payload, src, numel);
        }
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }

    if (!out) {
        std::cerr << "[GGUFWriter] write failed for " << out_path << "\n";
        return false;
    }
    std::cout << "[GGUFWriter] wrote " << outs.size() << " tensors"
              << (skipped ? " (" + std::to_string(skipped) + " unmapped skipped)" : "") << " -> " << out_path << "\n";
    return true;
}

}  // namespace io
}  // namespace serius
