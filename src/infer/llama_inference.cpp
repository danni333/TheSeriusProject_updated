#include "llama_inference.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../blas/blas.h"

namespace serius {
namespace infer {

using core::Tensor;

// ---- Helper: RMS Norm (in-place on output) ----
static void rms_norm(const float *input, const float *weight,
                     float *output, size_t rows, size_t dim, float eps)
{
    for (size_t r = 0; r < rows; ++r) {
        const float *in_row = input + r * dim;
        float *out_row = output + r * dim;
        float sum_sq = 0.0F;
        for (size_t i = 0; i < dim; ++i) {
            sum_sq += in_row[i] * in_row[i];
        }
        float rms = 1.0F / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
        for (size_t i = 0; i < dim; ++i) {
            out_row[i] = in_row[i] * rms * weight[i];
        }
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
    for (size_t i = 0; i < n; ++i) {
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
                               const Tensor &token_ids)
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

        // --- Multi-head attention with GQA ---
        // For each head: scores = q_h @ k_h^T * scale, then softmax, then context = probs @ v_h
        float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        size_t rep = n_heads / n_kv_heads;

        for (size_t head = 0; head < n_heads; ++head) {
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
                    // Apply scale + causal mask
                    float val = dot * scale;
                    if (ki > qi) { val = -1e9F; }  // causal mask
                    sc[qi * seq_len + ki] = val;
                }
            }

            // Softmax each row
            softmax_rows_inplace(sc, seq_len, seq_len);

            // Context = probs @ V_head  -> [seq_len, head_dim]
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

        // gate_up = silu(gate) * up
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
                                  int token_id, [[maybe_unused]] size_t pos)
{
    Tensor ids({1});
    ids[0] = static_cast<float>(token_id);
    // For now delegate — Phase 2 KV-cache will make this truly O(1) per token
    return llama_forward_inference(model, ids);
}

}  // namespace infer
}  // namespace serius
