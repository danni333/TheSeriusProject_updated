#include "semantic_hash.h"

#include <cmath>

namespace serius {
namespace knowledge {

SemanticHash::SemanticHash(size_t embedding_dim, size_t hash_bits, uint32_t seed)
    : embedding_dim_(embedding_dim), hash_bits_(hash_bits), rng_(seed)
{
    initialize_hyperplanes();
}

void SemanticHash::initialize_hyperplanes()
{
    hyperplanes_.clear();
    hyperplanes_.reserve(hash_bits_);

    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < hash_bits_; ++i) {
        core::Tensor hyperplane({embedding_dim_});

        // Generate random vector
        for (size_t j = 0; j < embedding_dim_; ++j) {
            hyperplane[j] = dist(rng_);
        }

        // Normalize to unit length
        float norm = 0.0f;
        for (size_t j = 0; j < embedding_dim_; ++j) {
            norm += hyperplane[j] * hyperplane[j];
        }
        norm = std::sqrt(norm);

        if (norm > 1e-6f) {
            for (size_t j = 0; j < embedding_dim_; ++j) {
                hyperplane[j] /= norm;
            }
        }

        hyperplanes_.push_back(hyperplane);
    }
}

uint64_t SemanticHash::hash(const core::Tensor &embedding) const
{
    if (embedding.size() != embedding_dim_) {
        return 0;
    }

    uint64_t h = 0;
    for (size_t i = 0; i < hash_bits_; ++i) {
        // Compute dot product
        float dot = 0.0f;
        const core::Tensor &plane = hyperplanes_[i];
        for (size_t j = 0; j < embedding_dim_; ++j) {
            dot += embedding[j] * plane[j];
        }

        // Set bit if dot product is positive
        if (dot > 0.0f) {
            h |= (1ULL << i);
        }
    }

    return h;
}

std::vector<uint64_t> SemanticHash::hash_batch(const std::vector<core::Tensor> &embeddings) const
{
    std::vector<uint64_t> hashes;
    hashes.reserve(embeddings.size());

    for (const auto &emb : embeddings) {
        hashes.push_back(hash(emb));
    }

    return hashes;
}

size_t SemanticHash::hamming_distance(uint64_t h1, uint64_t h2)
{
    uint64_t xor_result = h1 ^ h2;
    size_t count = 0;

    // Count set bits (Brian Kernighan's algorithm)
    while (xor_result) {
        xor_result &= (xor_result - 1);
        count++;
    }

    return count;
}

bool SemanticHash::is_similar(uint64_t h1, uint64_t h2, size_t max_distance)
{
    return hamming_distance(h1, h2) <= max_distance;
}

}  // namespace knowledge
}  // namespace serius
