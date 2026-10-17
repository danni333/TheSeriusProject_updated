#pragma once

#include <cstddef>

#include "../tensor/tensor.h"

namespace serius {
namespace model {

using core::Tensor;

constexpr float DEFAULT_ROPE_BASE = 10000.0F;

/// Computes the effective RoPE base when extending beyond the training context length.
/// For NTK/YaRN-style scaling, target_ctx > train_ctx implies a larger base so that
/// the rotation frequency decays more slowly over longer positions.
float compute_rope_base(float base, size_t d_head, size_t training_ctx = 0, size_t target_ctx = 0);

/// Applies Rotary Position Embedding (RoPE) in-place to Queries and Keys.
/// Expects Q and K to be of shape (seq_len, d_model) or (seq_len, d_head) where d = last dim.
/// Generates frequencies on the fly based on the sequence index.
void apply_rope(Tensor &q, Tensor &k, size_t d_head,
                float rope_base = DEFAULT_ROPE_BASE,
                size_t pos_offset = 0,
                size_t training_ctx = 0,
                size_t target_ctx = 0);

}  // namespace model
}  // namespace serius
