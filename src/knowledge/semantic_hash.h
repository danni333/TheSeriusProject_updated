#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace knowledge {

/// Semantic Locality-Sensitive Hashing for embeddings.
/// Maps similar semantic inputs to similar hash values using random hyperplanes.
class SemanticHash
{
   public:
    explicit SemanticHash(size_t embedding_dim, size_t hash_bits = 32, uint32_t seed = 42);

    /// Compute semantic hash from embedding vector.
    uint64_t hash(const core::Tensor &embedding) const;

    /// Compute hash for multiple embeddings.
    std::vector<uint64_t> hash_batch(const std::vector<core::Tensor> &embeddings) const;

    /// Compute Hamming distance between two hashes (number of differing bits).
    static size_t hamming_distance(uint64_t h1, uint64_t h2);

    /// Check if two hashes are similar (within a threshold).
    static bool is_similar(uint64_t h1, uint64_t h2, size_t max_distance = 4);

    size_t hash_bits() const { return hash_bits_; }
    size_t embedding_dim() const { return embedding_dim_; }

    /// Get the hyperplanes for inspection/testing.
    const std::vector<core::Tensor> &hyperplanes() const { return hyperplanes_; }

   private:
    size_t embedding_dim_;
    size_t hash_bits_;
    std::vector<core::Tensor> hyperplanes_;
    std::mt19937 rng_;

    /// Initialize random hyperplanes (unit vectors in embedding space).
    void initialize_hyperplanes();
};

}  // namespace knowledge
}  // namespace serius
