#pragma once

#include <memory>

#include "../autograd/node.h"
#include "../moe/moe_layer.h"
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
    TransformerBlock(size_t d_model, size_t n_heads, size_t d_ff, unsigned seed = 42,
                     size_t num_experts = 0, size_t top_k = 0, size_t shared_experts = 1,
                     bool enable_moe = false);

    /// Forward pass (pre-norm architecture)
    /// Input: (seq_len, d_model) or (batch_size, seq_len, d_model)
    Node *forward(Node *input, PagedKVCache *cache = nullptr) const;

    /// Optimized forward pass using fused kernels (e.g. fused_bias_gelu in FFN)
    Node *forward_optimized(Node *input, PagedKVCache *cache = nullptr) const;

    bool has_moe() const { return moe_layer_ != nullptr; }
    std::vector<Node *> moe_parameters() const;

    // Sub-layers
    LayerNorm norm1;
    MultiHeadAttention attention;
    LayerNorm norm2;
    std::unique_ptr<FFN> ffn;

    size_t d_model_;

   private:
    std::unique_ptr<MoELayer> moe_layer_;
};

}  // namespace model
}  // namespace serius
