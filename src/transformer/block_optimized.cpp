#include "../autograd/ops.h"
#include "block.h"


namespace serius {
namespace model {

using core::autograd::Node;

Node *TransformerBlock::forward_optimized(Node *input, KVCache *cache) const
{
    using namespace core::autograd::ops;

    // Implementation of optimized transformer block.
    // Standard forward is:
    //   normed1 = norm1(input)
    //   attn_out = attn(normed1)
    //   res1 = input + attn_out
    //   normed2 = norm2(res1)
    //   ffn_out = ffn(normed2)
    //   out = res1 + ffn_out
    //
    // Optimized forward fusions:
    // 1. fused_add_layernorm for Pre-Norm 2
    // 2. ffn.forward_optimized() combining linear and fused_bias_gelu

    // Pre-Norm 1: x + Attention(LayerNorm(x))
    // We don't have a fused layernorm without residual, so we use standard layernorm
    Node *normed1 = norm1.forward(input);
    Node *attn_out = attention.forward(normed1, cache);

    // Instead of res1 = input + attn_out AND normed2 = norm2(res1),
    // we use the fused_add_layernorm which computes res1 and normed2 in one pass
    // fused_add_layernorm(x, residual, gamma, beta) -> LayerNorm(x + residual)
    // Note: It returns the layernormed result. But we ALSO need res1 for the final addition!
    // The current fused_add_layernorm op returns only LayerNorm(x + residual).
    // If we need res1, we might have to do it separately unless we modify the fused op
    // to return both... Let's just do standard add for res1, OR we can use the
    // standard Layernorm on res1.
    // Looking at ops.h, fused_add_layernorm returns just Node*.
    // So if we use it, we lose `res1` which is needed for the final residual.
    // For now, let's just use FFN optimized forward.

    Node *res1 = add(input, attn_out);

    Node *normed2 = norm2.forward(res1);

    // Optimized FFN pass (fuses bias and gelu)
    Node *ffn_out = ffn.forward_optimized(normed2);

    // Final residual connection
    return add(res1, ffn_out);
}

}  // namespace model
}  // namespace serius
