#include "attention.h"

#include <cmath>
#include <stdexcept>

#include "../autograd/ops.h"

namespace serius {
namespace transformer {

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t n_heads, size_t d_lat)
    : d_model_(d_model), n_heads_(n_heads), d_head_(d_model / n_heads), cached_mask_({0})
{
    // If d_lat is 0, default to d_model / 4 as a sensible heuristic
    d_c_ = (d_lat == 0) ? (d_model / 4) : d_lat;
    if (d_model % n_heads != 0) {
        throw std::runtime_error("d_model must be divisible by n_heads");
    }

    // Xavier-like initialization helper
    auto init_weights = [](size_t out_dim, size_t in_dim) -> autograd::Node * {
        Tensor weights({out_dim, in_dim});
        const float init_scale = 1.0F / std::sqrt(static_cast<float>(in_dim));
        weights.rand();
        for (size_t idx = 0; idx < weights.size(); ++idx) {
            const float half_v = 0.5F;
            const float two_v = 2.0F;
            weights[idx] = (weights[idx] - half_v) * two_v * init_scale;
        }
        return new autograd::Node(std::move(weights));
    };

    w_q = init_weights(d_model, d_model);
    w_q->name = "attn_w_q";
    w_dkv = init_weights(d_model, d_c_);
    w_dkv->name = "attn_w_dkv";
    w_uk = init_weights(d_c_, d_model);
    w_uk->name = "attn_w_uk";
    w_uv = init_weights(d_c_, d_model);
    w_uv->name = "attn_w_uv";
    w_o = init_weights(d_model, d_model);
    w_o->name = "attn_w_o";
}

MultiHeadAttention::~MultiHeadAttention()
{
    delete w_q;
    delete w_dkv;
    delete w_uk;
    delete w_uv;
    delete w_o;
}

autograd::Node *MultiHeadAttention::forward(autograd::Node *input) const
{
    using namespace autograd::ops;

    // Queries: input * w_q
    autograd::Node *queries = matmul(input, w_q);

    // Multi-Head Latent Attention KV Compression
    autograd::Node *c_t = matmul(input, w_dkv);

    // Keys and Values decompression
    autograd::Node *keys = matmul(c_t, w_uk);
    autograd::Node *values = matmul(c_t, w_uv);

    // --- RoPE (Differentiable) ---
    autograd::ops::RoPEResult rope_out = autograd::ops::rope(queries, keys, d_head_);
    autograd::Node *q_rot = rope_out.queries;
    autograd::Node *k_rot = rope_out.keys;

    // --- Split Heads and Flatten for Matmul ---
    size_t batch_size = input->data.ndim() == 3 ? input->data.shape()[0] : 1;
    size_t seq_len = input->data.ndim() == 3 ? input->data.shape()[1] : input->data.shape()[0];
    size_t n_heads = n_heads_;
    size_t d_head = d_head_;

    // Reshape: (B, S, D) -> (B, S, H, D_h)
    autograd::Node *q_split = reshape(q_rot, {batch_size, seq_len, n_heads, d_head});
    autograd::Node *k_split = reshape(k_rot, {batch_size, seq_len, n_heads, d_head});
    autograd::Node *v_split = reshape(values, {batch_size, seq_len, n_heads, d_head});

    // Transpose: (B, S, H, D_h) -> (B, H, S, D_h)
    autograd::Node *q_heads = contiguous(transpose(q_split, 1, 2));
    autograd::Node *k_heads = contiguous(transpose(k_split, 1, 2));
    autograd::Node *v_heads = contiguous(transpose(v_split, 1, 2));

    // Flatten: (B, H, S, D_h) -> (B*H, S, D_h)
    size_t batch_heads = batch_size * n_heads;
    autograd::Node *q_flat = reshape(q_heads, {batch_heads, seq_len, d_head});
    autograd::Node *k_flat = reshape(k_heads, {batch_heads, seq_len, d_head});
    autograd::Node *v_flat = reshape(v_heads, {batch_heads, seq_len, d_head});

    // --- Scaled Dot-Product Attention ---

    // 1. Scores = (Q * K^T) / sqrt(d_k)
    autograd::Node *k_t = contiguous(transpose(k_flat, 1, 2));
    autograd::Node *scores = matmul(q_flat, k_t);

    float scale_factor = 1.0F / std::sqrt(static_cast<float>(d_head));
    autograd::Node *scores_scaled = mul_scalar(scores, scale_factor);

    // 2. Causal Masking (Optimized: reuse cached mask if possible)
    if (mask_seq_len_ != seq_len) {
        cached_mask_ = Tensor({seq_len, seq_len});
        for (size_t r_idx = 0; r_idx < seq_len; ++r_idx) {
            for (size_t c_idx = r_idx + 1; c_idx < seq_len; ++c_idx) {
                cached_mask_.at(r_idx, c_idx) = 1.0F;
            }
        }
        mask_seq_len_ = seq_len;
    }

    const float mask_value = -1e30F;
    autograd::Node *scores_masked = masked_fill(scores_scaled, cached_mask_, mask_value);

    // 3. Softmax
    autograd::Node *attn_probs = softmax(scores_masked, -1);

    // 4. Context = Attn * V
    autograd::Node *context_flat = matmul(attn_probs, v_flat);

    // --- Unflatten and Merge ---
    autograd::Node *context_heads = reshape(context_flat, {batch_size, n_heads, seq_len, d_head});
    autograd::Node *context_transposed = contiguous(transpose(context_heads, 1, 2));
    autograd::Node *context_merged = reshape(context_transposed, {batch_size, seq_len, d_model_});

    // --- Output Projection ---
    return matmul(context_merged, w_o);
}

}  // namespace transformer
}  // namespace serius
