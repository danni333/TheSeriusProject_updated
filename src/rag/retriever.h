#pragma once

#include <string>
#include <vector>

#include "embeddings.h"
#include "vector_client.h"
#include "nlohmann/json.hpp"

namespace serius {
namespace rag {

using json = nlohmann::json;

/// RAG Retriever: complete pipeline for retrieval-augmented generation.
class RAGRetriever
{
public:
    struct Config {
        size_t embedding_dim = 768;
        int top_k = 5;
        float similarity_threshold = 0.7f;
        std::string system_prompt = "Rispondi basandoti SOLO sui documenti forniti. Cita le fonti.";
    };

    explicit RAGRetriever(const Config& config = Config{});
    ~RAGRetriever();

    /// Add a document to the knowledge base.
    /// @param id Document ID
    /// @param text Document text
    /// @param metadata Additional metadata
    void add_document(const std::string& id, const std::string& text,
                      const json& metadata = json{});

    /// Retrieve relevant documents for a query.
    /// @param query User query
    /// @return RAG prompt with context
    std::string retrieve(const std::string& query);

    /// Build RAG prompt from query and retrieved documents.
    /// @param query User query
    /// @param docs Retrieved documents
    /// @return Formatted prompt
    std::string build_rag_prompt(const std::string& query,
                                 const std::vector<SearchResult>& docs) const;

    /// Get the underlying vector DB client.
    VectorDBClient& get_vector_db() { return vector_db_; }

    /// Clear all documents.
    void clear();

private:
    Config config_;
    VectorDBClient vector_db_;
};

}  // namespace rag
}  // namespace serius
