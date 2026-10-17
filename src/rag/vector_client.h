#pragma once

#include <string>
#include <vector>
#include <functional>

#include "../tensor/tensor.h"
#include "nlohmann/json.hpp"

namespace serius {
namespace rag {

using core::Tensor;
using json = nlohmann::json;

/// Result from a vector database search.
struct SearchResult {
    std::string id;
    std::string text;
    float score;
    json metadata;

    SearchResult() : score(0.0f) {}
};

/// Client for vector database operations.
/// Supports local in-memory storage and can be extended for remote DBs.
class VectorDBClient
{
public:
    struct Config {
        std::string endpoint;  // For remote DBs (e.g., Qdrant, Milvus)
        bool use_local = true;  // Use in-memory storage
        size_t dimension = 768;  // Embedding dimension
    };

    explicit VectorDBClient(const Config& config = Config{});
    ~VectorDBClient();

    /// Search for similar documents.
    /// @param query_embedding Query vector
    /// @param top_k Number of results to return
    /// @param threshold Minimum similarity score
    /// @return List of search results
    std::vector<SearchResult> search(const std::vector<float>& query_embedding,
                                      int top_k = 5, float threshold = 0.7f);

    /// Upsert (insert or update) a document.
    /// @param id Document ID
    /// @param embedding Document embedding
    /// @param text Document text
    /// @param metadata Additional metadata
    void upsert(const std::string& id, const std::vector<float>& embedding,
                const std::string& text, const json& metadata = json{});

    /// Delete a document by ID.
    /// @param id Document ID
    void remove(const std::string& id);

    /// Get document by ID.
    /// @param id Document ID
    /// @return Document (empty if not found)
    SearchResult get(const std::string& id) const;

    /// Clear all documents.
    void clear();

    /// Get number of documents.
    size_t size() const { return documents_.size(); }

private:
    Config config_;

    // In-memory storage
    struct Document {
        std::string id;
        std::vector<float> embedding;
        std::string text;
        json metadata;
    };

    std::vector<Document> documents_;

    // Helper: calculate cosine similarity
    float cosine_similarity(const std::vector<float>& a,
                           const std::vector<float>& b) const;
};

}  // namespace rag
}  // namespace serius
