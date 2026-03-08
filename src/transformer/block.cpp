#include "block.h"

#include "../autograd/ops.h"

namespace serius {
namespace transformer {

TransformerBlock::TransformerBlock(size_t d_model, size_t n_heads, size_t d_ff)
    : norm1(d_model), attention(d_model, n_heads), norm2(d_model), ffn(d_model, d_ff), d_model_(d_model)
{
}

autograd::Node *TransformerBlock::forward(autograd::Node *input, KVCache *cache) const
{
    using namespace autograd::ops;

    // Pre-Norm: x + Attention(LayerNorm(x))
    autograd::Node *normed1 = norm1.forward(input);
    autograd::Node *attn_out = attention.forward(normed1, cache);

    // Residual connection
    autograd::Node *residual1 = add(input, attn_out);

    // Pre-Norm: x + FFN(LayerNorm(x))
    autograd::Node *normed2 = norm2.forward(residual1);
    autograd::Node *ffn_out = ffn.forward(normed2);

    // Residual connection
    return add(residual1, ffn_out);
}

}  // namespace transformer
}  // namespace serius
