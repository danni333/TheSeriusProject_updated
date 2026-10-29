#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "../tensor/tensor.h"
#include "semantic_hash.h"

namespace serius {
namespace knowledge {

/// Compressed knowledge entry for a semantic hash bucket.
struct CompressedKnowledge
{
    core::Tensor latent;              // Compressed representation of the knowledge
    float confidence = 1.0f;          // Reliability score
    std::vector<std::string> sources; // Source documents
    size_t fact_count = 1;           // Number of facts merged into this entry
};

/// Atomic fact extracted from a document.
struct Fact
{
    std::string text;
    std::string source;
    float importance = 1.0f;
};

/// Knowledge Function: maps semantic hashes to compressed knowledge.
/// Provides O(1) knowledge lookup instead of O(log n) database search.
class KnowledgeFunction
{
   public:
    explicit KnowledgeFunction(size_t embedding_dim, size_t hash_bits = 32, uint32_t seed = 42);

    /// Build knowledge table from a corpus of documents.
    /// @param facts Atomic facts extracted from documents
    /// @param embeddings Pre-computed embeddings for each fact
    void build(const std::vector<Fact> &facts, const std::vector<core::Tensor> &embeddings);

    /// Query the knowledge function for a question.
    /// @param question_embedding Embedding of the query
    /// @return Compressed knowledge entry (empty if not found)
    const CompressedKnowledge *query(const core::Tensor &question_embedding) const;

    /// Query with fallback to lower resolution (smaller hash bits).
    const CompressedKnowledge *query_with_fallback(const core::Tensor &question_embedding,
                                                    size_t fallback_levels = 2) const;

    /// Merge two compressed knowledge entries.
    static CompressedKnowledge merge(const CompressedKnowledge &a, const CompressedKnowledge &b);

    /// Get the underlying semantic hasher.
    const SemanticHash &hasher() const { return hasher_; }

    /// Get the knowledge table (for inspection/testing).
    const std::unordered_map<uint64_t, CompressedKnowledge> &table() const { return table_; }

    /// Clear the knowledge table.
    void clear() { table_.clear(); }

    /// Get statistics about the knowledge table.
    struct Stats
    {
        size_t total_entries = 0;
        size_t total_facts = 0;
        float avg_confidence = 0.0f;
        size_t collisions = 0;  // Hash buckets with multiple facts
    };
    Stats get_stats() const;

   private:
    SemanticHash hasher_;
    std::unordered_map<uint64_t, CompressedKnowledge> table_;
    size_t embedding_dim_;

    /// Merge latent representations by averaging.
    static core::Tensor merge_latents(const core::Tensor &a, const core::Tensor &b);
};

}  // namespace knowledge
}  // namespace serius
