#include "../autograd/ops.h"
#include "../debug/tensor_logger.h"
#include "../hal/fused_kernels.h"
#include "block.h"

namespace serius {
namespace model {

// Optimized version with kernel fusion
Node *TransformerBlock::forward_optimized(Node *input, PagedKVCache *cache) const
{
    using namespace core::autograd::ops;
    using namespace serius::debug;

    // Initialize tensor logger
    TensorLogger &logger = get_tensor_logger();

    // 1. LayerNorm + Residual (fused se possibile)
    Node *normed1 = norm1.forward(input);
    logger.log(normed1->tensor(), "block_norm1", 0);
    Node *attn_out = attention.forward(normed1, cache);
    logger.log(attn_out->tensor(), "block_attn", 0);

    // 2. Fused Add + LayerNorm
    Node *normed_fused = fused_add_layernorm(input, attn_out, norm2.gamma_, norm2.beta_);
    logger.log(normed_fused->tensor(), "block_fused_ln", 0);

    // 3. FFN (potrebbe essere fusionato)
    Node *ffn_out_opt = ffn.forward_optimized(normed_fused);
    logger.log(ffn_out_opt->tensor(), "block_ffn", 0);

    // 4. Final residual skip
    Node *res_intermediate = add(input, attn_out);
    logger.log(res_intermediate->tensor(), "block_residual", 0);
    return add(res_intermediate, ffn_out_opt);
}

}  // namespace model
}  // namespace serius