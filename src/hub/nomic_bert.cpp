#include "nomic_bert.h"

#include "../blas/blas.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace serius::hub {

namespace {

constexpr float kSilkEps = 1e-12f;

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

/// Transpose a row-major [rows, cols] matrix in place-equivalent fashion:
/// returns a new vector holding the [cols, rows] layout so that
/// y = x @ W^T can run as a single GEMM (x:[T,in], Wt:[in,out] -> [T,out]).
std::vector<float> transpose_matrix(const std::vector<float> &src, size_t rows, size_t cols)
{
    std::vector<float> dst(src.size());
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    return dst;
}

/// y[t,o] = sum_i x[t,i] * wt[i*out + o]  — single GEMM via core::blas (AVX2/512).
void linear_gemm(const float *x, const float *wt, float *y, int T, int in, int out)
{
    // Non-owning tensor views over caller buffers; blas::matmul never writes A/B.
    core::Tensor A(const_cast<float *>(x), {static_cast<size_t>(T), static_cast<size_t>(in)});
    core::Tensor B(const_cast<float *>(wt), {static_cast<size_t>(in), static_cast<size_t>(out)});
    core::Tensor C(y, {static_cast<size_t>(T), static_cast<size_t>(out)});
    core::blas::matmul(A, B, C);
    // blas::matmul may dispatch to CUDA asynchronously; pull the result back
    // into the caller's host buffer before returning.
    C.syncToHost();
}

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

bool NomicBertEmbedder::get_tensor_transposed(
    const std::unordered_map<std::string, core::Tensor> &tensors, const std::string &name,
    size_t out_dim, size_t in_dim, std::vector<float> *out)
{
    std::vector<float> flat;
    if (!get_tensor(tensors, name, &flat) || flat.size() != out_dim * in_dim) {
        return false;
    }
    *out = transpose_matrix(flat, out_dim, in_dim);
    return true;
}

bool NomicBertEmbedder::load(const std::unordered_map<std::string, core::Tensor> &tensors)
{
    return load(tensors, Config{});
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
        // Weights are stored transposed ([in,out]) so the forward pass runs
        // as plain GEMMs: y = x @ Wt.
        if (!get_tensor_transposed(tensors, p + "attn_qkv.weight", 3 * H, H, &w.qkv)) return false;
        if (!get_tensor_transposed(tensors, p + "attn_output.weight", H, H, &w.wo)) return false;
        if (!get_tensor_transposed(tensors, p + "ffn_up.weight", F, H, &w.up)) return false;
        if (!get_tensor_transposed(tensors, p + "ffn_gate.weight", F, H, &w.gate)) return false;
        if (!get_tensor_transposed(tensors, p + "ffn_down.weight", H, F, &w.down)) return false;
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
    // RoPE inv_freq: theta^(-2i/hd), i in [0, hd/2). cos/sin tables are
    // position-only — compute once and reuse across all layers.
    const int half = hd / 2;
    std::vector<float> rope_cos(static_cast<size_t>(T) * half);
    std::vector<float> rope_sin(static_cast<size_t>(T) * half);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < half; ++i) {
            const float ang = t * std::pow(cfg_.rope_theta, -2.0f * i / hd);
            rope_cos[static_cast<size_t>(t) * half + i] = std::cos(ang);
            rope_sin[static_cast<size_t>(t) * half + i] = std::sin(ang);
        }
    }

    std::vector<float> qkv(static_cast<size_t>(T) * 3 * H);
    std::vector<float> attn(static_cast<size_t>(T) * H);
    std::vector<float> proj(static_cast<size_t>(T) * H);
    std::vector<float> ff1(static_cast<size_t>(T) * F);
    std::vector<float> ff2(static_cast<size_t>(T) * F);
    std::vector<float> ffu(static_cast<size_t>(T) * F);
    std::vector<float> ffz(static_cast<size_t>(T) * H);

    for (const LayerW &w : layers_) {
        // QKV projection (GEMM)
        linear_gemm(x.data(), w.qkv.data(), qkv.data(), T, H, 3 * H);

        // RoPE on Q and K (NeoX rotate-half), per head per position
        auto rope = [&](int base_off, int t) {
            const float *cs = rope_cos.data() + static_cast<size_t>(t) * half;
            const float *sn = rope_sin.data() + static_cast<size_t>(t) * half;
            for (int h = 0; h < nh; ++h) {
                float *head = qkv.data() + static_cast<size_t>(t) * 3 * H + base_off + h * hd;
                for (int i = 0; i < half; ++i) {
                    const float c = cs[i], s = sn[i];
                    const float x1 = head[i], x2 = head[i + half];
                    head[i] = x1 * c - x2 * s;
                    head[i + half] = x2 * c + x1 * s;
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
        linear_gemm(attn.data(), w.wo.data(), proj.data(), T, H, H);
        for (int t = 0; t < T; ++t) {
            float *xr = x.data() + static_cast<size_t>(t) * H;
            const float *pr = proj.data() + static_cast<size_t>(t) * H;
            for (int d = 0; d < H; ++d) xr[d] += pr[d];
            layernorm(xr, w.attn_norm_w.data(), w.attn_norm_b.data(), H, cfg_.norm_eps);
        }

        // SwiGLU FFN: down( silu(gate(x)) * up(x) ) + residual + post-norm
        linear_gemm(x.data(), w.up.data(), ffu.data(), T, H, F);
        linear_gemm(x.data(), w.gate.data(), ff1.data(), T, H, F);
        for (size_t i = 0; i < ffu.size(); ++i) ffu[i] *= silu(ff1[i]);
        linear_gemm(ffu.data(), w.down.data(), ffz.data(), T, F, H);
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
