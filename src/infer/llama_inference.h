#pragma once

#include "../tensor/tensor.h"
#include "../transformer/llama_model.h"

namespace serius {
namespace infer {

/// Inference-only LLaMA forward pass — no autograd overhead.
/// Takes token IDs and produces logits using direct Tensor operations.
///
/// @param model  Loaded LLaMAModel (weights are read from its Nodes' .data)
/// @param token_ids  1D tensor of token IDs, shape [seq_len]
/// @return Logits tensor, shape [seq_len, vocab_size]
core::Tensor llama_forward_inference(const model::LLaMAModel &model,
                                     const core::Tensor &token_ids);

/// Same but for a single token (optimized path for generation steps).
/// @param model  Loaded LLaMAModel
/// @param token_id  Single token ID
/// @param pos  Position offset for RoPE
/// @return Logits tensor, shape [1, vocab_size]
core::Tensor llama_forward_single_token(const model::LLaMAModel &model,
                                        int token_id, size_t pos);

}  // namespace infer
}  // namespace serius
