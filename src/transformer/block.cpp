#include "block.h"

#include "../autograd/ops.h"

namespace serius {
namespace model {

using core::autograd::Node;

TransformerBlock::TransformerBlock(size_t d_model, size_t n_heads, size_t d_ff)
    : norm1(d_model), attention(d_model, n_heads), norm2(d_model), ffn(d_model, d_ff), d_model_(d_model)
{
}

Node *TransformerBlock::forward(Node *input, KVCache *cache) const
{
    using namespace core::autograd::ops;

    // Pre-Norm: x + Attention(LayerNorm(x))
    Node *normed1 = norm1.forward(input);
    Node *attn_out = attention.forward(normed1, cache);

    // Residual connection
    Node *residual1 = add(input, attn_out);

    // Pre-Norm: x + FFN(LayerNorm(x))
    Node *normed2 = norm2.forward(residual1);
    Node *ffn_out = ffn.forward(normed2);

    // Residual connection
    return add(residual1, ffn_out);
}

}  // namespace model
}  // namespace serius
