#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "../autograd/node.h"
#include "../tensor/tensor.h"


namespace serius {
namespace model {
namespace attention {

// Default block sizes for Flash Attention tuning
namespace {
constexpr int DEFAULT_BLOCK_M = 128;
constexpr int DEFAULT_BLOCK_N = 128;
}  // namespace

struct FlashAttentionConfig
{
    bool is_causal = true;
    float softmax_scale = 1.0F;
    // Block size defaults that can be adjusted for tuning performance
    int block_m = DEFAULT_BLOCK_M;  // Sequence length block size for Query
    int block_n = DEFAULT_BLOCK_N;  // Sequence length block size for Key/Value
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
