#pragma once

#include "../autograd/node.h"
#include "../tensor/tensor.h"
#include "paged_kv_cache.h"

namespace serius {
namespace model {

using core::Tensor;
using core::autograd::Node;

class PagedKVCache;

class MultiHeadAttention
{
   public:
    MultiHeadAttention(size_t d_model, size_t n_heads, size_t d_lat = 0);
    ~MultiHeadAttention();

    /// Returns: (seq_len, d_model)
    Node *forward(Node *input, PagedKVCache *cache = nullptr) const;

    // Weight matrices for Multi-Head Latent Attention (MLA)
    Node *w_q;    // (d_model, d_model)
    Node *w_dkv;  // down-proj: (d_model, d_c)
    Node *w_uk;   // up-proj K: (d_c, d_model)
    Node *w_uv;   // up-proj V: (d_c, d_model)
    Node *w_o;    // output proj: (d_model, d_model)

   private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_head_;
    size_t d_c_;  // compressed latent dimension
    mutable Tensor cached_mask_;
    mutable size_t mask_seq_len_ = 0;
};

}  // namespace model
}  // namespace serius
