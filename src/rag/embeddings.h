#pragma once

#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace rag {

using core::Tensor;

/// Compute embedding for text using mean pooling.
/// In a production system, this would use a dedicated embedding model.
/// For now, we provide a simple hash-based embedding for testing.
///
/// @param text Input text
/// @param dimension Embedding dimension
/// @return Normalized embedding vector
std::vector<float> compute_embedding(const std::string& text, size_t dimension = 768);

/// Normalize embedding to unit L2 norm.
/// @param embedding Embedding vector (modified in-place)
void normalize_embedding(std::vector<float>& embedding);

/// Compute L2 norm of a vector.
/// @param vec Input vector
/// @return L2 norm
float l2_norm(const std::vector<float>& vec);

/// Compute cosine similarity between two embeddings.
/// @param a First embedding
/// @param b Second embedding
/// @return Cosine similarity in [-1, 1]
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

}  // namespace rag
}  // namespace serius
