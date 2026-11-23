#pragma once

#include <functional>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace rag {

using core::Tensor;

/// Embedding backend: maps text to a vector. Used by RAGRetriever so real
/// models (EmbeddingManager, NomicBert, remote /v1/embeddings, ...) can be
/// plugged in instead of the built-in hash fallback.
using EmbedFn = std::function<std::vector<float>(const std::string &text)>;

/// Compute embedding for text using a deterministic hash projection.
/// This is a fallback for tests/offline use — it carries no semantic
/// signal. Production callers should install a real embedder via
/// RAGRetriever::set_embedder (EmbeddingManager, NomicBert, remote
/// /v1/embeddings, ...).
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
