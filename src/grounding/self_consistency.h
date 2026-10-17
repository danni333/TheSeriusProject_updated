#pragma once

#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace grounding {

using core::Tensor;

/// Cosine similarity between two vectors (0 = orthogonal, 1 = identical).
float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b);

/// Self-consistency: pick the response with the highest average similarity
/// to all other responses. Returns the index of the most consistent response.
size_t self_consistent_index(const std::vector<std::vector<float>> &embeddings);

/// Majority voting for categorical responses (e.g. short answers).
std::string majority_vote(const std::vector<std::string> &responses);

}  // namespace grounding
}  // namespace serius
