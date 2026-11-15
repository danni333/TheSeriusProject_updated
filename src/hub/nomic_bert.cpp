#include "nomic_bert.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace serius::hub {

namespace {

constexpr float kSilkEps = 1e-12f;

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

/// LayerNorm over `n` contiguous floats: out = (x-mean)/sqrt(var+eps)*w + b.
void layernorm(float *x, const float *w, const float *b, int n, float eps)
{
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
    for (int i = 0; i < n; ++i) {
        x[i] = (x[i] - static_cast<float>(mean)) * inv * w[i] + b[i];
    }
}

/// y[t,o] = sum_i x[t,i] * w[o*in + i]   (w is [out,in] row-major)
void linear_rows(const float *x, const float *w, float *y, int T, int in, int out)
{
    for (int t = 0; t < T; ++t) {
        const float *xr = x + static_cast<size_t>(t) * in;
        float *yr = y + static_cast<size_t>(t) * out;
        for (int o = 0; o < out; ++o) {
            const float *wr = w + static_cast<size_t>(o) * in;
            float acc = 0.0f;
            for (int i = 0; i < in; ++i) acc += xr[i] * wr[i];
            yr[o] = acc;
        }
    }
}

}  // namespace

void NomicBertEmbedder::set_vocab(const std::vector<std::string> &tokens)
{
    vocab_.clear();
    vocab_.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        vocab_.emplace(tokens[i], static_cast<int>(i));
    }
}

bool NomicBertEmbedder::get_tensor(const std::unordered_map<std::string, core::Tensor> &tensors,
                                   const std::string &name, std::vector<float> *out)
{
    auto it = tensors.find(name);
    if (it == tensors.end() || it->second.size() == 0) {
        return false;
    }
    const size_t n = it->second.size();
    out->resize(n);
    std::memcpy(out->data(), it->second.data(), n * sizeof(float));
    return true;
}

bool NomicBertEmbedder::load(const std::unordered_map<std::string, core::Tensor> &tensors,
                             const Config &cfg)
{
    cfg_ = cfg;
    const int H = cfg_.hidden;
    const int F = cfg_.ffn;

    if (!get_tensor(tensors, "token_embd.weight", &tok_embd_)) return false;
    if (tok_embd_.size() != static_cast<size_t>(vocab_.size()) * H && !vocab_.empty()) {
        // vocab not set yet or mismatch — accept anyway if size is a multiple of H
        if (tok_embd_.size() % H != 0) return false;
    }
    if (!get_tensor(tensors, "token_types.weight", &type_embd_row0_)) {
        type_embd_row0_.assign(H, 0.0f);
    } else if (type_embd_row0_.size() > static_cast<size_t>(H)) {
        // ggml order (H, n_types): row 0 occupies first H floats
        type_embd_row0_.resize(H);
    }
    if (!get_tensor(tensors, "token_embd_norm.weight", &tok_norm_w_)) return false;
    if (!get_tensor(tensors, "token_embd_norm.bias", &tok_norm_b_)) return false;

    layers_.assign(cfg_.n_layers, LayerW{});
    for (int l = 0; l < cfg_.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        LayerW &w = layers_[l];
        if (!get_tensor(tensors, p + "attn_qkv.weight", &w.qkv)) return false;
        if (!get_tensor(tensors, p + "attn_output.weight", &w.wo)) return false;
        if (!get_tensor(tensors, p + "ffn_up.weight", &w.up)) return false;
        if (!get_tensor(tensors, p + "ffn_gate.weight", &w.gate)) return false;
        if (!get_tensor(tensors, p + "ffn_down.weight", &w.down)) return false;
        if (!get_tensor(tensors, p + "attn_output_norm.weight", &w.attn_norm_w)) return false;
        if (!get_tensor(tensors, p + "attn_output_norm.bias", &w.attn_norm_b)) return false;
        if (!get_tensor(tensors, p + "layer_output_norm.weight", &w.out_norm_w)) return false;
        if (!get_tensor(tensors, p + "layer_output_norm.bias", &w.out_norm_b)) return false;
        // sanity: expected flat sizes
        if (w.qkv.size() != static_cast<size_t>(3 * H) * H ||
            w.wo.size() != static_cast<size_t>(H) * H ||
            w.up.size() != static_cast<size_t>(F) * H ||
            w.gate.size() != static_cast<size_t>(F) * H ||
            w.down.size() != static_cast<size_t>(H) * F) {
            return false;
        }
    }
    loaded_ = true;
    return true;
}

// ---------------- WordPiece (BERT uncased) ----------------

namespace {

bool is_wordpiece_punct(char c)
{
    // BERT basic tokenization: ASCII punctuation splits tokens.
    const unsigned char u = static_cast<unsigned char>(c);
    return (u >= 33 && u <= 47) || (u >= 58 && u <= 64) ||
           (u >= 91 && u <= 96) || (u >= 123 && u <= 126);
}

}  // namespace

std::vector<int> NomicBertEmbedder::tokenize(const std::string &text) const
{
    std::vector<int> ids;
    ids.reserve(64);
    ids.push_back(cfg_.cls_id);

    if (vocab_.empty()) {
        ids.push_back(cfg_.sep_id);
        return ids;
    }

    // Basic tokenization: lowercase, split on whitespace and punctuation.
    std::vector<std::string> words;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) { words.push_back(cur); cur.clear(); }
    };
    for (char ch : text) {
        const unsigned char u = static_cast<unsigned char>(ch);
        if (u < 128) {
            const char c = static_cast<char>(std::tolower(u));
            if (std::isspace(u) || is_wordpiece_punct(c)) {
                flush();
                if (!std::isspace(u)) words.emplace_back(1, c);
            } else {
                cur.push_back(c);
            }
        } else {
            cur.push_back(ch);  // keep non-ASCII bytes inside the word
        }
    }
    flush();

    for (const std::string &word : words) {
        // Greedy longest-match-first WordPiece.
        size_t start = 0;
        bool failed = false;
        while (start < word.size()) {
            size_t best = word.size();
            std::string best_tok;
            for (size_t end = word.size(); end > start; --end) {
                std::string cand = (start == 0 ? "" : "##") + word.substr(start, end - start);
                if (vocab_.find(cand) != vocab_.end()) {
                    best = end;
                    best_tok = cand;
                    break;
                }
            }
            if (best == word.size() && vocab_.find(word.substr(start)) == vocab_.end()) {
                failed = true;
                break;
            }
            if (best_tok.empty()) { failed = true; break; }
            ids.push_back(vocab_.at(best_tok));
            start = best;
        }
        if (failed) ids.push_back(cfg_.unk_id);
    }

    ids.push_back(cfg_.sep_id);
    if (static_cast<int>(ids.size()) > cfg_.max_len) {
        ids.resize(cfg_.max_len);
        ids.back() = cfg_.sep_id;
    }
    return ids;
}

// ---------------- Forward ----------------

std::vector<float> NomicBertEmbedder::forward_hidden(const std::vector<int> &ids) const
{
    const int H = cfg_.hidden;
    const int F = cfg_.ffn;
    const int nh = cfg_.n_heads;
    const int hd = H / nh;
    const int T = static_cast<int>(ids.size());
    const int vocab = static_cast<int>(tok_embd_.size() / H);

    // 1. Embedding lookup + token-type row 0 + LayerNorm
    std::vector<float> x(static_cast<size_t>(T) * H);
    for (int t = 0; t < T; ++t) {
        const int id = (ids[t] >= 0 && ids[t] < vocab) ? ids[t] : cfg_.unk_id;
        const float *e = tok_embd_.data() + static_cast<size_t>(id) * H;
        float *xr = x.data() + static_cast<size_t>(t) * H;
        for (int d = 0; d < H; ++d) xr[d] = e[d] + type_embd_row0_[d];
        layernorm(xr, tok_norm_w_.data(), tok_norm_b_.data(), H, cfg_.norm_eps);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    // RoPE inv_freq: theta^(-2i/hd), i in [0, hd/2)
    std::vector<float> inv_freq(hd / 2);
    for (int i = 0; i < hd / 2; ++i) {
        inv_freq[i] = std::pow(cfg_.rope_theta, -2.0f * i / hd);
    }

    std::vector<float> qkv(static_cast<size_t>(T) * 3 * H);
    std::vector<float> attn(static_cast<size_t>(T) * H);
    std::vector<float> proj(static_cast<size_t>(T) * H);
    std::vector<float> ff1(static_cast<size_t>(T) * F);
    std::vector<float> ff2(static_cast<size_t>(T) * F);
    std::vector<float> ffu(static_cast<size_t>(T) * F);
    std::vector<float> ffz(static_cast<size_t>(T) * H);

    for (const LayerW &w : layers_) {
        // QKV projection
        linear_rows(x.data(), w.qkv.data(), qkv.data(), T, H, 3 * H);

        // RoPE on Q and K (NeoX rotate-half), per head per position
        auto rope = [&](int base_off, int t) {
            for (int h = 0; h < nh; ++h) {
                float *head = qkv.data() + static_cast<size_t>(t) * 3 * H + base_off + h * hd;
                for (int i = 0; i < hd / 2; ++i) {
                    const float ang = t * inv_freq[i];
                    const float c = std::cos(ang), s = std::sin(ang);
                    const float x1 = head[i], x2 = head[i + hd / 2];
                    head[i] = x1 * c - x2 * s;
                    head[i + hd / 2] = x2 * c + x1 * s;
                }
            }
        };
        for (int t = 0; t < T; ++t) {
            rope(0, t);                    // Q at offset 0
            rope(H, t);                    // K at offset H
        }

        // Attention (non-causal, mean over all positions)
        for (int h = 0; h < nh; ++h) {
            for (int t = 0; t < T; ++t) {
                const float *q = qkv.data() + static_cast<size_t>(t) * 3 * H + h * hd;
                std::vector<float> scores(T);
                float mx = -1e30f;
                for (int s = 0; s < T; ++s) {
                    const float *k = qkv.data() + static_cast<size_t>(s) * 3 * H + H + h * hd;
                    float acc = 0.0f;
                    for (int d = 0; d < hd; ++d) acc += q[d] * k[d];
                    scores[s] = acc * scale;
                    mx = std::max(mx, scores[s]);
                }
                float sum = 0.0f;
                for (int s = 0; s < T; ++s) {
                    scores[s] = std::exp(scores[s] - mx);
                    sum += scores[s];
                }
                float *out = attn.data() + static_cast<size_t>(t) * H + h * hd;
                for (int d = 0; d < hd; ++d) out[d] = 0.0f;
                for (int s = 0; s < T; ++s) {
                    const float p = scores[s] / sum;
                    const float *v = qkv.data() + static_cast<size_t>(s) * 3 * H + 2 * H + h * hd;
                    for (int d = 0; d < hd; ++d) out[d] += p * v[d];
                }
            }
        }

        // Output projection + residual + post-norm
        linear_rows(attn.data(), w.wo.data(), proj.data(), T, H, H);
        for (int t = 0; t < T; ++t) {
            float *xr = x.data() + static_cast<size_t>(t) * H;
            const float *pr = proj.data() + static_cast<size_t>(t) * H;
            for (int d = 0; d < H; ++d) xr[d] += pr[d];
            layernorm(xr, w.attn_norm_w.data(), w.attn_norm_b.data(), H, cfg_.norm_eps);
        }

        // SwiGLU FFN: down( silu(gate(x)) * up(x) ) + residual + post-norm
        linear_rows(x.data(), w.up.data(), ffu.data(), T, H, F);
        linear_rows(x.data(), w.gate.data(), ff1.data(), T, H, F);
        for (size_t i = 0; i < ffu.size(); ++i) ffu[i] *= silu(ff1[i]);
        linear_rows(ffu.data(), w.down.data(), ffz.data(), T, F, H);
        for (int t = 0; t < T; ++t) {
            float *xr = x.data() + static_cast<size_t>(t) * H;
            const float *zr = ffz.data() + static_cast<size_t>(t) * H;
            for (int d = 0; d < H; ++d) xr[d] += zr[d];
            layernorm(xr, w.out_norm_w.data(), w.out_norm_b.data(), H, cfg_.norm_eps);
        }
    }
    return x;
}

std::vector<float> NomicBertEmbedder::embed(const std::string &text) const
{
    if (!loaded_) return {};
    const auto ids = tokenize(text);
    if (ids.size() < 2) return {};
    const auto hidden = forward_hidden(ids);
    const int H = cfg_.hidden;
    const int T = static_cast<int>(ids.size());

    // Mean pooling over all positions (pooling_type=1 in GGUF metadata)
    std::vector<float> emb(H, 0.0f);
    for (int t = 0; t < T; ++t) {
        const float *r = hidden.data() + static_cast<size_t>(t) * H;
        for (int d = 0; d < H; ++d) emb[d] += r[d];
    }
    double norm = 0.0;
    for (auto &v : emb) {
        v /= static_cast<float>(T);
        norm += static_cast<double>(v) * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (auto &v : emb) v = static_cast<float>(v / norm);
    }
    return emb;
}

}  // namespace serius::hub
