// EXPERIMENTAL: modulo didattico/sperimentale — vedi docs/experimental.md

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "fact_extractor.h"
#include "knowledge_fn.h"

namespace serius {
namespace knowledge {

/// End-to-end knowledge base: documents → FactExtractor → embeddings →
/// KnowledgeFunction (LSH-compressed lookup).
///
/// This is the missing glue between the standalone knowledge components:
/// callers supply a real embedding backend (EmbeddingManager, NomicBert,
/// remote /v1/embeddings, ...) via set_embedder; without one, add_document
/// and query report an error through the return values below.
class KnowledgeBase
{
   public:
    /// Embedder: text → vector of exactly `embedding_dim` floats.
    /// Same signature as rag::EmbedFn; any compatible std::function works.
    using EmbedFn = std::function<std::vector<float>(const std::string &text)>;

    /// Result of a query: the compressed entry plus the fact texts stored
    /// under the exact hash bucket (empty for fallback-resolution hits).
    struct QueryResult
    {
        const CompressedKnowledge *entry = nullptr;
        std::vector<std::string> fact_texts;
        bool found = false;
    };

    explicit KnowledgeBase(size_t embedding_dim, size_t hash_bits = 32, uint32_t seed = 42);

    /// Install the embedding backend. Must be called before add_document/query.
    void set_embedder(EmbedFn fn) { embedder_ = std::move(fn); }
    bool has_embedder() const { return static_cast<bool>(embedder_); }

    /// Fact extraction configuration passthrough.
    FactExtractor &extractor() { return extractor_; }
    const FactExtractor &extractor() const { return extractor_; }

    /// Extract facts from `text` and index them. Returns the number of facts
    /// actually embedded and stored (0 on embedder/dimension errors).
    size_t add_document(const std::string &text, const std::string &source = "unknown");

    /// Query with an embedded question. `fallback_levels` mirrors
    /// KnowledgeFunction::query_with_fallback.
    QueryResult query(const std::string &question, size_t fallback_levels = 2) const;

    /// Access the underlying function/table for inspection.
    const KnowledgeFunction &function() const { return kf_; }
    KnowledgeFunction::Stats stats() const { return kf_.get_stats(); }
    void clear();

   private:
    /// Embed + L2-normalize; empty vector on failure.
    std::vector<float> embed(const std::string &text) const;

    FactExtractor extractor_;
    KnowledgeFunction kf_;
    EmbedFn embedder_;
    size_t embedding_dim_;

    /// fact texts keyed by the exact semantic hash of their embedding.
    std::unordered_map<uint64_t, std::vector<std::string>> fact_texts_;
};

}  // namespace knowledge
}  // namespace serius
