#pragma once

#include "../autograd/node.h"
#include "../tensor/tensor.h"

namespace serius {
namespace transformer {

class MultiHeadAttention
{
   public:
    MultiHeadAttention(size_t d_model, size_t n_heads, size_t d_lat = 0);
    ~MultiHeadAttention();

    /// Forward pass
    /// Input: (seq_len, d_model)
    /// Returns: (seq_len, d_model)
    autograd::Node *forward(autograd::Node *input) const;

    // Weight matrices for Multi-Head Latent Attention (MLA)
    autograd::Node *w_q;    // (d_model, d_model)
    autograd::Node *w_dkv;  // down-proj: (d_model, d_c)
    autograd::Node *w_uk;   // up-proj K: (d_c, d_model)
    autograd::Node *w_uv;   // up-proj V: (d_c, d_model)
    autograd::Node *w_o;    // output proj: (d_model, d_model)

   private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_head_;
    size_t d_c_;  // compressed latent dimension
    mutable Tensor cached_mask_;
    mutable size_t mask_seq_len_ = 0;
};

}  // namespace transformer
}  // namespace serius
