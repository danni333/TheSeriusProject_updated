#include "llama_inference.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../blas/blas.h"

namespace serius {
namespace infer {

using core::Tensor;

#include <immintrin.h>

// Helper: Horizontal sum of AVX registers
inline float _mm256_reduce_add_ps(__m256 x) {
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(sum);
}

// ---- Helper: RMS Norm (in-place on output) ----
static void rms_norm(const float *input, const float *weight,
                     float *output, size_t rows, size_t dim, float eps)
{
    for (size_t r = 0; r < rows; ++r) {
        const float *in_row = input + r * dim;
        float *out_row = output + r * dim;
        
        float sum_sq = 0.0F;
        size_t i = 0;
        
        // SIMD Sum of Squares
        __m256 vsum = _mm256_setzero_ps();
        for (; i + 7 < dim; i += 8) {
            __m256 v = _mm256_loadu_ps(in_row + i);
            vsum = _mm256_fmadd_ps(v, v, vsum);
        }
        sum_sq = _mm256_reduce_add_ps(vsum);
        for (; i < dim; ++i) sum_sq += in_row[i] * in_row[i];
        
        float rms = 1.0F / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
        __m256 vrms = _mm256_set1_ps(rms);
        
        for (i = 0; i + 7 < dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(in_row + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            __m256 vo = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vw);
            _mm256_storeu_ps(out_row + i, vo);
        }
        for (; i < dim; ++i) out_row[i] = in_row[i] * rms * weight[i];
    }
}

// ---- Helper: Apply RoPE in-place ----
static void apply_rope_inplace(float *data, size_t seq_len, size_t d_model,
                               size_t d_head, float rope_base, size_t pos_offset)
{
    for (size_t p = 0; p < seq_len; ++p) {
        size_t abs_pos = p + pos_offset;
        float *row = data + p * d_model;
        for (size_t h_off = 0; h_off < d_model; h_off += d_head) {
            for (size_t i = 0; i < d_head; i += 2) {
                float freq = 1.0F / std::pow(rope_base,
                    static_cast<float>(i) / static_cast<float>(d_head));
                float theta = static_cast<float>(abs_pos) * freq;
                float cos_v = std::cos(theta);
                float sin_v = std::sin(theta);
                float x0 = row[h_off + i];
                float x1 = row[h_off + i + 1];
                row[h_off + i]     = x0 * cos_v - x1 * sin_v;
                row[h_off + i + 1] = x0 * sin_v + x1 * cos_v;
            }
        }
    }
}

// ---- Helper: SiLU(x) = x * sigmoid(x) ----
static void silu_inplace(float *data, size_t n)
{
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(data + i);
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, vx);
        for(int k=0; k<8; ++k) {
            float x = tmp[k];
            float s = (x >= 0.0F) ? (1.0F / (1.0F + std::exp(-x)))
                                 : (std::exp(x) / (1.0F + std::exp(x)));
            tmp[k] = x * s;
        }
        _mm256_storeu_ps(data + i, _mm256_load_ps(tmp));
    }
    for (; i < n; ++i) {
        float x = data[i];
        float s = (x >= 0.0F) ? (1.0F / (1.0F + std::exp(-x)))
                               : (std::exp(x) / (1.0F + std::exp(x)));
        data[i] = x * s;
    }
}

// ---- Helper: Softmax over last dimension (in-place) ----
static void softmax_rows_inplace(float *data, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; ++r) {
        float *row = data + r * cols;
        float max_val = *std::max_element(row, row + cols);
        float sum = 0.0F;
        for (size_t c = 0; c < cols; ++c) {
            row[c] = std::exp(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0F / sum;
        for (size_t c = 0; c < cols; ++c) {
            row[c] *= inv_sum;
        }
    }
}

// ---- Main inference function ----
Tensor llama_forward_inference(const model::LLaMAModel &model,
                               const Tensor &token_ids,
                               KVCache *cache)
{
    const auto &cfg = model.config();
    const size_t seq_len = token_ids.size();
    const size_t hidden = cfg.hidden_size;
    const size_t n_heads = cfg.num_attention_heads;
    const size_t n_kv_heads = cfg.num_key_value_heads;
    const size_t head_dim = hidden / n_heads;
    const size_t kv_dim = n_kv_heads * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_base = cfg.rope_theta;

    // ---- Embedding lookup ----
    Tensor h({seq_len, hidden});
    const float *embed = model.embed_tokens()->data.data();
    for (size_t i = 0; i < seq_len; ++i) {
        size_t tid = static_cast<size_t>(token_ids[i]);
        if (tid >= cfg.vocab_size) { tid = 0; }
        const float *src = embed + tid * hidden;
        float *dst = h.data() + i * hidden;
        std::copy(src, src + hidden, dst);
    }

    // Pre-allocate reusable scratch buffers
    Tensor normed({seq_len, hidden});
    Tensor q_buf({seq_len, n_heads * head_dim});
    Tensor k_buf({seq_len, kv_dim});
    Tensor v_buf({seq_len, kv_dim});
    Tensor attn_concat({seq_len, n_heads * head_dim});
    Tensor mlp_gate({seq_len, inter});
    Tensor mlp_up({seq_len, inter});
    Tensor mlp_down({seq_len, hidden});

    const auto &layers = model.layers();
    if (cache && cache->empty()) {
        cache->layers.resize(layers.size());
        for (auto &cl : cache->layers) {
            cl.k = Tensor({4096, kv_dim});
            cl.v = Tensor({4096, kv_dim});
            cl.current_len = 0;
        }
    }

    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto *blk = layers[layer_idx];

        // --- Pre-attention RMSNorm ---
        rms_norm(h.data(), blk->input_layernorm_weight->data.data(),
                 normed.data(), seq_len, hidden, eps);

        // --- QKV projections: matmul(normed, weight) ---
        core::blas::matmul(normed, blk->q_proj->data, q_buf);
        core::blas::matmul(normed, blk->k_proj->data, k_buf);
        core::blas::matmul(normed, blk->v_proj->data, v_buf);

        // --- RoPE ---
        apply_rope_inplace(q_buf.data(), seq_len, n_heads * head_dim,
                           head_dim, rope_base, 0);
        apply_rope_inplace(k_buf.data(), seq_len, kv_dim,
                           head_dim, rope_base, 0);

        // --- KV Cache Update ---
        if (cache) {
            KVCacheLayer &cl = cache->layers[layer_idx];
            // If cache current_len is 0, we are processing the initial prompt
            size_t start_pos = cl.current_len;
            if (start_pos + seq_len <= 4096) {
                for (size_t i = 0; i < seq_len; ++i) {
                    std::copy(k_buf.data() + i * kv_dim, k_buf.data() + (i + 1) * kv_dim,
                              cl.k.data() + (start_pos + i) * kv_dim);
                    std::copy(v_buf.data() + i * kv_dim, v_buf.data() + (i + 1) * kv_dim,
                              cl.v.data() + (start_pos + i) * kv_dim);
                }
                cl.current_len += seq_len;
            }
        }

        // --- Multi-head attention with GQA ---
        float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        size_t rep = n_heads / n_kv_heads;

        #pragma omp parallel for
        for (int head = 0; head < static_cast<int>(n_heads); ++head) {
            size_t kv_head = head / rep;

            // Build score matrix [seq_len, seq_len]
            // score[i][j] = sum_d q[i, head, d] * k[j, kv_head, d]
            Tensor scores({seq_len, seq_len});
            float *sc = scores.data();

            for (size_t qi = 0; qi < seq_len; ++qi) {
                const float *q_row = q_buf.data() + qi * n_heads * head_dim + head * head_dim;
                for (size_t ki = 0; ki < seq_len; ++ki) {
                    const float *k_row = k_buf.data() + ki * kv_dim + kv_head * head_dim;
                    float dot = 0.0F;
                    for (size_t d = 0; d < head_dim; ++d) {
                        dot += q_row[d] * k_row[d];
                    }
                    float val = dot * scale;
                    if (ki > qi) { val = -1e9F; }  // causal mask
                    sc[qi * seq_len + ki] = val;
                }
            }

            softmax_rows_inplace(sc, seq_len, seq_len);

            for (size_t qi = 0; qi < seq_len; ++qi) {
                float *out = attn_concat.data() + qi * n_heads * head_dim + head * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0F;
                    for (size_t vi = 0; vi < seq_len; ++vi) {
                        sum += sc[qi * seq_len + vi]
                            * (v_buf.data()[vi * kv_dim + kv_head * head_dim + d]);
                    }
                    out[d] = sum;
                }
            }
        }

        // --- O projection ---
        Tensor o_out({seq_len, hidden});
        core::blas::matmul(attn_concat, blk->o_proj->data, o_out);

        // --- Residual 1 ---
        float *h_ptr = h.data();
        const float *o_ptr = o_out.data();
        for (size_t i = 0; i < seq_len * hidden; ++i) {
            h_ptr[i] += o_ptr[i];
        }

        // --- Post-attention RMSNorm ---
        rms_norm(h.data(), blk->post_attention_layernorm_weight->data.data(),
                 normed.data(), seq_len, hidden, eps);

        // --- SwiGLU MLP ---
        core::blas::matmul(normed, blk->gate_proj->data, mlp_gate);
        core::blas::matmul(normed, blk->up_proj->data, mlp_up);

        silu_inplace(mlp_gate.data(), seq_len * inter);

        float *gate_p = mlp_gate.data();
        const float *up_p = mlp_up.data();
        for (size_t i = 0; i < seq_len * inter; ++i) {
            gate_p[i] *= up_p[i];
        }

        core::blas::matmul(mlp_gate, blk->down_proj->data, mlp_down);

        // --- Residual 2 ---
        const float *down_p = mlp_down.data();
        for (size_t i = 0; i < seq_len * hidden; ++i) {
            h_ptr[i] += down_p[i];
        }
    }

    // --- Final RMSNorm ---
    rms_norm(h.data(), model.norm_weight()->data.data(),
             normed.data(), seq_len, hidden, eps);

    // --- LM Head: logits = normed @ lm_head ---
    Tensor logits({seq_len, cfg.vocab_size});
    core::blas::matmul(normed, model.lm_head()->data, logits);

    return logits;
}

// ---- Single-token optimized path ----
Tensor llama_forward_single_token(const model::LLaMAModel &model,
                                  int token_id, size_t pos,
                                  KVCache *cache)
{
    const auto &cfg = model.config();
    const size_t hidden = cfg.hidden_size;
    const size_t n_heads = cfg.num_attention_heads;
    const size_t n_kv_heads = cfg.num_key_value_heads;
    const size_t head_dim = hidden / n_heads;
    const size_t kv_dim = n_kv_heads * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_base = cfg.rope_theta;

    // --- Embedding lookup (single token) ---
    const float *embed = model.embed_tokens()->data.data();
    size_t tid = static_cast<size_t>(token_id);
    if (tid >= cfg.vocab_size) tid = 0;

    Tensor h_local({1, hidden});
    if (cache && cache->scratch_h.size() == 0) cache->scratch_h = Tensor({1, hidden});
    Tensor &h = (cache) ? cache->scratch_h : h_local;
    std::copy(embed + tid * hidden, embed + (tid + 1) * hidden, h.data());

    // Initialize scratch buffers in cache if needed
    if (cache) {
        if (cache->scratch_normed.size() == 0) cache->scratch_normed = Tensor({1, hidden});
        if (cache->scratch_q.size() == 0) cache->scratch_q = Tensor({1, n_heads * head_dim});
        if (cache->scratch_k.size() == 0) cache->scratch_k = Tensor({1, kv_dim});
        if (cache->scratch_v.size() == 0) cache->scratch_v = Tensor({1, kv_dim});
        if (cache->scratch_attn_concat.size() == 0) cache->scratch_attn_concat = Tensor({1, n_heads * head_dim});
        if (cache->scratch_mlp_gate.size() == 0) cache->scratch_mlp_gate = Tensor({1, inter});
        if (cache->scratch_mlp_up.size() == 0) cache->scratch_mlp_up = Tensor({1, inter});
        if (cache->scratch_mlp_down.size() == 0) cache->scratch_mlp_down = Tensor({1, hidden});
        if (cache->scratch_scores.size() < n_heads * 4096) cache->scratch_scores.resize(n_heads * 4096);
    }

    // Local fallbacks if no cache
    Tensor normed_local({1, hidden}), q_local({1, n_heads * head_dim}), k_local({1, kv_dim}), v_local({1, kv_dim});
    Tensor attn_local({1, n_heads * head_dim}), gate_local({1, inter}), up_local({1, inter}), down_local({1, hidden});
    std::vector<float> scores_local;

    Tensor &normed = (cache) ? cache->scratch_normed : normed_local;
    Tensor &q_buf = (cache) ? cache->scratch_q : q_local;
    Tensor &k_buf = (cache) ? cache->scratch_k : k_local;
    Tensor &v_buf = (cache) ? cache->scratch_v : v_local;
    Tensor &attn_concat = (cache) ? cache->scratch_attn_concat : attn_local;
    Tensor &mlp_gate = (cache) ? cache->scratch_mlp_gate : gate_local;
    Tensor &mlp_up = (cache) ? cache->scratch_mlp_up : up_local;
    Tensor &mlp_down = (cache) ? cache->scratch_mlp_down : down_local;

    const auto &layers = model.layers();
    if (cache && cache->layers.empty()) {
        cache->layers.resize(layers.size());
        for (auto &cl : cache->layers) {
            cl.k = Tensor({4096, kv_dim}); 
            cl.v = Tensor({4096, kv_dim});
            cl.current_len = 0;
        }
    }

    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto *blk = layers[layer_idx];

        // --- Pre-attention RMSNorm ---
        rms_norm(h.data(), blk->input_layernorm_weight->data.data(),
                 normed.data(), 1, hidden, eps);

        // --- QKV projections ---
        core::blas::matmul(normed, blk->q_proj->data, q_buf);
        core::blas::matmul(normed, blk->k_proj->data, k_buf);
        core::blas::matmul(normed, blk->v_proj->data, v_buf);

        // --- RoPE (at current pos) ---
        apply_rope_inplace(q_buf.data(), 1, n_heads * head_dim, head_dim, rope_base, pos);
        apply_rope_inplace(k_buf.data(), 1, kv_dim, head_dim, rope_base, pos);

        // --- KV Cache Update ---
        size_t effective_len = pos + 1;
        if (cache) {
            KVCacheLayer &cl = cache->layers[layer_idx];
            std::copy(k_buf.data(), k_buf.data() + kv_dim, cl.k.data() + pos * kv_dim);
            std::copy(v_buf.data(), v_buf.data() + kv_dim, cl.v.data() + pos * kv_dim);
            cl.current_len = effective_len;
        }

        // --- Multi-head attention ---
        float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        size_t rep = n_heads / n_kv_heads;

        #pragma omp parallel for
        for (int head_int = 0; head_int < static_cast<int>(n_heads); ++head_int) {
            size_t head = static_cast<size_t>(head_int);
            size_t kv_head = head / rep;
            
            float* scores_ptr;
            if (cache) {
                scores_ptr = cache->scratch_scores.data() + (head * 4096);
            } else {
                scores_ptr = scores_local.data();
            }
            const float *q_row = q_buf.data() + head * head_dim;

            for (size_t j = 0; j < effective_len; ++j) {
                const float *k_row = (cache) ? (cache->layers[layer_idx].k.data() + j * kv_dim + kv_head * head_dim) 
                                             : (k_buf.data() + kv_head * head_dim);
                float dot = 0.0F;
                for (size_t d = 0; d < head_dim; ++d) dot += q_row[d] * k_row[d];
                scores_ptr[j] = dot * scale;
            }

            // Softmax
            float max_val = -1e30F;
            for (size_t j = 0; j < effective_len; ++j) if (scores_ptr[j] > max_val) max_val = scores_ptr[j];
            
            float sum = 0.0F;
            for (size_t j = 0; j < effective_len; ++j) {
                scores_ptr[j] = std::exp(scores_ptr[j] - max_val);
                sum += scores_ptr[j];
            }
            float inv_sum = 1.0F / sum;
            for (size_t j = 0; j < effective_len; ++j) scores_ptr[j] *= inv_sum;

            // Context = sum_j scores[j] * V[j]
            float *out = attn_concat.data() + head * head_dim;
            std::fill(out, out + head_dim, 0.0F);
            for (size_t j = 0; j < effective_len; ++j) {
                const float *v_row = (cache) ? (cache->layers[layer_idx].v.data() + j * kv_dim + kv_head * head_dim)
                                             : (v_buf.data() + kv_head * head_dim);
                float s = scores_ptr[j];
                for (size_t d = 0; d < head_dim; ++d) out[d] += s * v_row[d];
            }
        }

        // --- O projection ---
        core::blas::matmul(attn_concat, blk->o_proj->data, mlp_down); // mlp_down reused for o_out

        // --- Residual 1 ---
        float *h_ptr = h.data();
        const float *o_ptr = mlp_down.data();
        for (size_t i = 0; i < hidden; ++i) h_ptr[i] += o_ptr[i];

        // --- Post-attention RMSNorm ---
        rms_norm(h.data(), blk->post_attention_layernorm_weight->data.data(),
                 normed.data(), 1, hidden, eps);

        // --- MLP ---
        core::blas::matmul(normed, blk->gate_proj->data, mlp_gate);
        core::blas::matmul(normed, blk->up_proj->data, mlp_up);
        silu_inplace(mlp_gate.data(), inter);
        float *gate_p = mlp_gate.data();
        const float *up_p = mlp_up.data();
        for (size_t i = 0; i < inter; ++i) gate_p[i] *= up_p[i];
        core::blas::matmul(mlp_gate, blk->down_proj->data, mlp_down);

        // --- Residual 2 ---
        const float *down_p = mlp_down.data();
        for (size_t i = 0; i < hidden; ++i) h_ptr[i] += down_p[i];
    }

    // --- Final RMSNorm ---
    rms_norm(h.data(), model.norm_weight()->data.data(), normed.data(), 1, hidden, eps);

    // --- LM Head ---
    Tensor logits({1, cfg.vocab_size}); 
    core::blas::matmul(normed, model.lm_head()->data, logits);

    return logits;
}

}  // namespace infer
}  // namespace serius
