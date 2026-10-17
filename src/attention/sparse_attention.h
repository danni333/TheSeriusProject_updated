#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace attention {

using core::Tensor;

/// Result of sparse attention computation.
struct SparseAttentionResult {
    Tensor output;                      // (seq_len, d_model)
    std::vector<std::vector<int>> active_keys;  // For each query: indices of active keys
    float sparsity_ratio;              // Fraction of attention skipped
};

/// Dynamic sparse attention (CPU implementation).
/// For each query token, selects only the most relevant key tokens based on a threshold.
///
/// @param Q Query tensor (seq_len, d_model)
/// @param K Key tensor (seq_len, d_model)
/// @param V Value tensor (seq_len, d_model)
/// @param threshold Score threshold for key selection
/// @param top_k Alternative: select top-k keys per query (0 = use threshold)
/// @return SparseAttentionResult with output and active key indices
SparseAttentionResult dynamic_sparse_attention(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    float threshold = 0.1f, size_t top_k = 0);

/// Calculate cosine similarity between two vectors.
float cosine_similarity(const float* a, const float* b, size_t dim);

/// Calculate dot product with scaling.
float scaled_dot_product(const float* a, const float* b, size_t dim, float scale);

}  // namespace attention
}  // namespace serius
