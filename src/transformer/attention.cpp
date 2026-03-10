#include "attention.h"

#include <cmath>
#include <stdexcept>

#include "../autograd/ops.h"
#include "paged_kv_cache.h"

namespace serius {
namespace model {

using core::autograd::Node;

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t n_heads, size_t d_lat)
    : d_model_(d_model), n_heads_(n_heads), d_head_(d_model / n_heads), cached_mask_({0})
{
    // If d_lat is 0, default to d_model / 4 as a sensible heuristic
    d_c_ = (d_lat == 0) ? (d_model / 4) : d_lat;
    if (d_model % n_heads != 0) {
        throw std::runtime_error("d_model must be divisible by n_heads");
    }

    // Xavier-like initialization helper
    auto init_weights = [](size_t out_dim, size_t in_dim) -> Node * {
        Tensor weights({out_dim, in_dim});
        const float init_scale = 1.0F / std::sqrt(static_cast<float>(in_dim));
        weights.rand();
        for (size_t idx = 0; idx < weights.size(); ++idx) {
            const float half_v = 0.5F;
            const float two_v = 2.0F;
            weights[idx] = (weights[idx] - half_v) * two_v * init_scale;
        }
        return new Node(std::move(weights));
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

Node *MultiHeadAttention::forward(Node *input, PagedKVCache *cache) const
{
    using namespace core::autograd::ops;

    // Queries: input * w_q
    Node *queries = matmul(input, w_q);

    // Multi-Head Latent Attention KV Compression
    Node *c_t = matmul(input, w_dkv);

    // Keys and Values decompression
    Node *keys = matmul(c_t, w_uk);
    Node *values = matmul(c_t, w_uv);

    // --- RoPE (Differentiable) ---
    size_t pos_offset = cache ? cache->current_len() : 0;
    RoPEResult rope_out = rope(queries, keys, d_head_, DEFAULT_ROPE_BASE, pos_offset);
    Node *q_rot = rope_out.queries;
    Node *k_rot = rope_out.keys;

    // --- Split Heads and Flatten for Matmul ---
    size_t batch_size = input->data.ndim() == 3 ? input->data.shape()[0] : 1;
    size_t seq_len = input->data.ndim() == 3 ? input->data.shape()[1] : input->data.shape()[0];
    size_t n_heads = n_heads_;
    size_t d_head = d_head_;

    // Reshape: (B, S, D) -> (B, S, H, D_h)
    Node *q_split = reshape(q_rot, {batch_size, seq_len, n_heads, d_head});
    Node *k_split = reshape(k_rot, {batch_size, seq_len, n_heads, d_head});
    Node *v_split = reshape(values, {batch_size, seq_len, n_heads, d_head});

    // Transpose: (B, S, H, D_h) -> (B, H, S, D_h)
    Node *q_heads = contiguous(transpose(q_split, 1, 2));
    Node *k_heads = contiguous(transpose(k_split, 1, 2));
    Node *v_heads = contiguous(transpose(v_split, 1, 2));

    // --- KV Caching Integration ---
    if (cache != nullptr) {
        // Update cache with CURRENT step rotated keys and values
        // k_heads: (B, H, S, D_h), v_heads: (B, H, S, D_h)
        cache->update(k_heads->data, v_heads->data);

        // Get FULL sequence from cache
        Tensor full_k = cache->gather_keys();
        Tensor full_v = cache->gather_values();

        // Create new nodes for cached data (gradient flow stops here for cache for now)
        k_heads = new Node(std::move(full_k));
        v_heads = new Node(std::move(full_v));

        k_heads = contiguous(k_heads);
        v_heads = contiguous(v_heads);
    }

    size_t total_seq_len = (cache != nullptr) ? cache->current_len() : seq_len;

    // Flatten: (B, H, S, D_h) -> (B*H, S, D_h)
    size_t batch_heads = batch_size * n_heads;
    Node *q_flat = reshape(q_heads, {batch_heads, seq_len, d_head});
    Node *k_flat = reshape(k_heads, {batch_heads, total_seq_len, d_head});
    Node *v_flat = reshape(v_heads, {batch_heads, total_seq_len, d_head});

    // --- Scaled Dot-Product Attention ---

    // 1. Scores = (Q * K^T) / sqrt(d_k)
    Node *k_t = contiguous(transpose(k_flat, 1, 2));
    Node *scores = matmul(q_flat, k_t);

    float scale_factor = 1.0F / std::sqrt(static_cast<float>(d_head));
    // 2. Causal Masking (Optimized: reuse cached mask if possible)
    // In incremental mode (cache provided), seq_len is typically 1, but we need full attn mask relative to
    // total_seq_len
    if (mask_seq_len_ != total_seq_len) {
        cached_mask_ = Tensor({total_seq_len, total_seq_len});
        for (size_t r_idx = 0; r_idx < total_seq_len; ++r_idx) {
            for (size_t c_idx = r_idx + 1; c_idx < total_seq_len; ++c_idx) {
                cached_mask_.at(r_idx, c_idx) = 1.0F;
            }
        }
        mask_seq_len_ = total_seq_len;
    }

    // Slice mask for current queries: [seq_len, total_seq_len]
    // If seq_len=1 (inference), we need row [pos_offset] from mask
    Tensor current_mask = cached_mask_.slice(0, pos_offset, pos_offset + seq_len);
    current_mask = current_mask.slice(1, 0, total_seq_len);  // [seq_len, total_seq_len]

    // 2. Fused Masked Softmax (scaling, masking, and softmax in one step)
    Node *attn_probs = fused_masked_softmax(scores, current_mask, scale_factor);

    // 4. Context = Attn * V
    Node *context_flat = matmul(attn_probs, v_flat);

    // --- Unflatten and Merge ---
    Node *context_heads = reshape(context_flat, {batch_size, n_heads, seq_len, d_head});
    Node *context_transposed = contiguous(transpose(context_heads, 1, 2));
    Node *context_merged = reshape(context_transposed, {batch_size, seq_len, d_model_});

    fprintf(stderr, "[Attention] Projecting output\n");
    fflush(stderr);
    // Final linear projection
    Node *output = matmul(context_merged, w_o);

    if (input->data.ndim() == 2) {
        fprintf(stderr, "[Attention] Reshaping to 2D\n");
        fflush(stderr);
        return reshape(output, {seq_len, d_model_});
    }
    return output;
}

}  // namespace model
}  // namespace serius
