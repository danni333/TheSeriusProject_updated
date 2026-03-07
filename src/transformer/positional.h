#pragma once

#include "../tensor/tensor.h"

namespace serius {
namespace transformer {

/// Applies Rotary Position Embedding (RoPE) in-place to Queries and Keys.
/// Expects Q and K to be of shape (seq_len, d_model) or (seq_len, d_head) where d = last dim.
/// Generates frequencies on the fly based on the sequence index.
void apply_rope(Tensor &q, Tensor &k, size_t d_head);

}  // namespace transformer
}  // namespace serius
