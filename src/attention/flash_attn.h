#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "../autograd/node.h"
#include "../tensor/tensor.h"


namespace serius {
namespace model {
namespace attention {

struct FlashAttentionConfig
{
    bool is_causal = true;
    float softmax_scale = 1.0F;
    // Forward tile sizes: zero selects the benchmarked profile; explicit values must be 16, 32, 64, or 128.
    int block_m = 0;
    int block_n = 0;
};

// Forward version for inference (no logsumexp returned)
core::Tensor flash_attention_forward(const core::Tensor &query, const core::Tensor &key, const core::Tensor &value,
                                     const FlashAttentionConfig &config);

// Internal version for autograd (returns {Output, LogSumExp})
std::pair<core::Tensor, core::Tensor> flash_attention_forward_internal(const core::Tensor &query,
                                                                       const core::Tensor &key,
                                                                       const core::Tensor &value,
                                                                       const FlashAttentionConfig &config);

// Autograd wrapper for Flash Attention
core::autograd::Node *flash_attention(core::autograd::Node *query, core::autograd::Node *key,
                                      core::autograd::Node *value, const FlashAttentionConfig &config);

}  // namespace attention
}  // namespace model
}  // namespace serius
