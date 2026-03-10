#include "block.h"

#include <iostream>

#include "../autograd/ops.h"

namespace serius {
namespace model {

using core::autograd::Node;

TransformerBlock::TransformerBlock(size_t d_model, size_t n_heads, size_t d_ff)
    : norm1(d_model), attention(d_model, n_heads), norm2(d_model), ffn(d_model, d_ff), d_model_(d_model)
{
}

Node *TransformerBlock::forward(Node *input, PagedKVCache *cache) const
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

Node *TransformerBlock::forward_optimized(Node *input, PagedKVCache *cache) const
{
    using namespace core::autograd::ops;
    std::cerr << "[Block] Start" << std::endl;

    // Use fused_add_layernorm where possible
    // x = x + Attn(LN(x))
    Node *normed1 = norm1.forward(input);
    std::cerr << "[Block] Norm1 done" << std::endl;
    Node *attn_out = attention.forward(normed1, cache);
    std::cerr << "[Block] Attention done" << std::endl;

    // x2 = x + attn_out
    // normed2 = LN(x2)
    // Fusion: normed2 = LN(x + attn_out)
    Node *normed_fused = fused_add_layernorm(input, attn_out, norm2.gamma_, norm2.beta_);
    std::cerr << "[Block] Fused Add LN done" << std::endl;

    // x3 = x2 + FFN(normed2)
    Node *ffn_out_opt = ffn.forward_optimized(normed_fused);
    std::cerr << "[Block] FFN done" << std::endl;

    // Final residual skip
    Node *res_intermediate = add(input, attn_out);
    return add(res_intermediate, ffn_out_opt);
}

}  // namespace model
}  // namespace serius
