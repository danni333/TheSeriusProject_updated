#pragma once

#include "../autograd/node.h"
#include "attention.h"
#include "ffn.h"
#include "layernorm.h"

namespace serius {
namespace model {

using core::autograd::Node;

class PagedKVCache;

class TransformerBlock
{
   public:
    TransformerBlock(size_t d_model, size_t n_heads, size_t d_ff);

    /// Forward pass (pre-norm architecture)
    /// Input: (seq_len, d_model) or (batch_size, seq_len, d_model)
    Node *forward(Node *input, PagedKVCache *cache = nullptr) const;

    /// Optimized forward pass using fused kernels (e.g. fused_bias_gelu in FFN)
    Node *forward_optimized(Node *input, PagedKVCache *cache = nullptr) const;

    // Sub-layers
    LayerNorm norm1;
    MultiHeadAttention attention;
    LayerNorm norm2;
    FFN ffn;

    size_t d_model_;
};

}  // namespace model
}  // namespace serius
