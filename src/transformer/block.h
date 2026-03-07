#pragma once

#include "../autograd/node.h"
#include "attention.h"
#include "ffn.h"
#include "layernorm.h"


namespace serius {
namespace transformer {

class TransformerBlock
{
   public:
    TransformerBlock(size_t d_model, size_t n_heads, size_t d_ff);

    /// Forward pass (pre-norm architecture)
    /// Input: (seq_len, d_model) or (batch_size, seq_len, d_model)
    autograd::Node *forward(autograd::Node *input) const;

    // Sub-layers
    LayerNorm norm1;
    MultiHeadAttention attention;
    LayerNorm norm2;
    FFN ffn;

    size_t d_model_;
};

}  // namespace transformer
}  // namespace serius
