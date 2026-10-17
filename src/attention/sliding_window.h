#pragma once

#include <cstddef>
#include <limits>

#include "../tensor/tensor.h"

namespace serius {
namespace attention {

using core::Tensor;

/// Creates a sliding window attention mask.
/// Each position i can only attend to positions in [max(0, i - window_size + 1), i].
/// The mask is filled with -INFINITY for disallowed positions and 0.0 for allowed ones.
///
/// @param seq_len Length of the sequence
/// @param window_size Size of the sliding window (how many positions back to attend)
/// @return Mask tensor of shape (seq_len, seq_len)
Tensor sliding_window_mask(size_t seq_len, size_t window_size);

/// Creates a combined global + local attention mask (Longformer-style).
/// Global tokens (indices in global_indices) can attend to all positions.
/// Local tokens attend only within their sliding window and to global tokens.
///
/// @param seq_len Length of the sequence
/// @param window_size Size of the sliding window for local attention
/// @param global_indices Indices of tokens with global attention
/// @return Mask tensor of shape (seq_len, seq_len)
Tensor global_local_mask(size_t seq_len, size_t window_size,
                         const std::vector<size_t>& global_indices);

/// Checks if the sliding window mask is correctly configured.
/// @param seq_len Sequence length
/// @param window_size Window size
/// @return true if valid (window_size > 0 and window_size <= seq_len)
bool is_valid_window(size_t seq_len, size_t window_size);

}  // namespace attention
}  // namespace serius
