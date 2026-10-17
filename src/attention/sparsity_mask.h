#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"

#ifdef SERIUS_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace serius {
namespace attention {

using core::Tensor;

/// Pattern types for block-sparse attention.
enum class SparsityPattern {
    CAUSAL,           // Standard causal pattern
    SLIDING_WINDOW,   // Sliding window pattern
    RANDOM,           // Random sparse pattern
    FIXED,            // Fixed pattern from layout
    LONGFORMER,       // Longformer-style global + local
    BIGBIRD           // BigBird-style random + global + local
};

/// Generate a block-sparse mask for attention.
/// @param seq_len Sequence length
/// @param block_size Size of each block (must divide seq_len)
/// @param pattern Type of sparsity pattern
/// @param window_size For sliding window patterns
/// @return Mask tensor of shape (num_blocks, num_blocks) where 1 = active, 0 = inactive
Tensor generate_block_mask(size_t seq_len, size_t block_size,
                          SparsityPattern pattern = SparsityPattern::CAUSAL,
                          size_t window_size = 0);

/// Generate a causal block mask (lower triangular).
Tensor causal_block_mask(size_t num_blocks);

/// Generate a sliding window block mask.
Tensor sliding_window_block_mask(size_t num_blocks, size_t window_blocks);

/// Generate a random sparse block mask with given sparsity ratio.
Tensor random_block_mask(size_t num_blocks, float sparsity_ratio);

/// Generate Longformer-style mask (global + local blocks).
Tensor longformer_block_mask(size_t num_blocks, size_t window_blocks,
                             const std::vector<size_t>& global_blocks);

/// Calculate sparsity ratio from a block mask.
float calculate_sparsity_ratio(const Tensor& block_mask);

/// Convert block mask to full attention mask.
/// @param block_mask Block-level mask (num_blocks, num_blocks)
/// @param block_size Size of each block
/// @return Full mask (seq_len, seq_len)
Tensor block_mask_to_full_mask(const Tensor& block_mask, size_t block_size);

#ifdef SERIUS_USE_CUDA
/// Block-sparse attention (CUDA implementation).
/// Only computes attention for active blocks specified by block_mask.
///
/// @param Q Query tensor (seq_len, d_model) on GPU
/// @param K Key tensor (seq_len, d_model) on GPU
/// @param V Value tensor (seq_len, d_model) on GPU
/// @param block_mask Block-level mask (num_blocks, num_blocks) on CPU
/// @param block_size Size of each block
/// @param stream CUDA stream
/// @return Output tensor (seq_len, d_model) on GPU
Tensor block_sparse_attention_cuda(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    const Tensor& block_mask, size_t block_size, cudaStream_t stream = 0);
#endif

}  // namespace attention
}  // namespace serius
