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
    size_t pos_offset = (cache != nullptr) ? cache->current_len() : 0;
    RoPEResult rope_out = rope(queries, keys, d_head_, DEFAULT_ROPE_BASE, pos_offset);
    Node *q_rot = rope_out.queries;
    Node *k_rot = rope_out.keys;

    // --- Split Heads and Flatten for Matmul ---
    size_t batch_size = input->data.ndim() == 3 ? input->data.shape()[0] : 1;
    size_t seq_len = input->data.ndim() == 3 ? input->data.shape()[1] : input->data.shape()[0];
    size_t n_heads = n_heads_;
    size_t d_head = d_head_;

    Node *q_split = reshape(q_rot, {batch_size, seq_len, n_heads, d_head});
    Node *k_split = reshape(k_rot, {batch_size, seq_len, n_heads, d_head});
    Node *v_split = reshape(values, {batch_size, seq_len, n_heads, d_head});

    Node *q_heads = contiguous(transpose(q_split, 1, 2));
    Node *k_heads = contiguous(transpose(k_split, 1, 2));
    Node *v_heads = contiguous(transpose(v_split, 1, 2));

    if (cache != nullptr) {
        cache->update(k_heads->data, v_heads->data);
        Tensor full_k = cache->gather_keys();
        Tensor full_v = cache->gather_values();
        k_heads = new Node(std::move(full_k));
        v_heads = new Node(std::move(full_v));
        k_heads = contiguous(k_heads);
        v_heads = contiguous(v_heads);
    }
    size_t total_seq_len = (cache != nullptr) ? cache->current_len() : seq_len;

    // Standard Path (CPU or fallbacks)
    // Scores: (B, H, S_q, D_h) * (B, H, D_h, S_k) -> (B, H, S_q, S_k)
    Node *k_heads_t = transpose(k_heads, 2, 3);
    Node *scores = matmul(q_heads, k_heads_t);

    float scale = 1.0F / std::sqrt(static_cast<float>(d_head));
    Node *probs;

    if (cache == nullptr) {
        // Training or prefill: apply causal mask
        if (mask_seq_len_ != total_seq_len) {
            cached_mask_ = core::Tensor({total_seq_len, total_seq_len});
            const float MASK_INIT_VAL = 0.0F; 
            cached_mask_.fill(MASK_INIT_VAL); 
            for (size_t i = 0; i < total_seq_len; ++i) {
                for (size_t j = 0; j < total_seq_len; ++j) {
                    cached_mask_.at(i, j) = (j > i) ? 1.0F : 0.0F;
                }
            }
            mask_seq_len_ = total_seq_len;
        }
        probs = fused_masked_softmax(scores, cached_mask_, scale);
    } else {
        probs = softmax(mul_scalar(scores, scale), -1);
    }

    // Context: (B, H, S_q, S_k) * (B, H, S_k, D_h) -> (B, H, S_q, D_h)
    Node *context_heads = matmul(probs, v_heads);

    // --- Unflatten and Merge ---
    Node *context_transposed = contiguous(transpose(context_heads, 1, 2));
    Node *context_merged = reshape(context_transposed, {batch_size, seq_len, d_model_});

    // Final linear projection
    Node *output = matmul(context_merged, w_o);

    if (input->data.ndim() == 2) {
        return reshape(output, {seq_len, d_model_});
    }
    return output;
}

}  // namespace model
}  // namespace serius
