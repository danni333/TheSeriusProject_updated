#include "retriever.h"

#include <sstream>

namespace serius {
namespace rag {

RAGRetriever::RAGRetriever()
    : RAGRetriever(Config{})
{
}

RAGRetriever::RAGRetriever(const Config& config)
    : config_(config), vector_db_(VectorDBClient::Config{.use_local = true, .dimension = config.embedding_dim})
{
}

RAGRetriever::~RAGRetriever()
{
}

void RAGRetriever::add_document(const std::string& id, const std::string& text,
                                const json& metadata)
{
    vector_db_.upsert(id, embed(text), text, metadata);
}

std::string RAGRetriever::retrieve(const std::string& query)
{
    std::vector<float> query_embedding = embed(query);
    std::vector<SearchResult> docs = vector_db_.search(query_embedding, config_.top_k,
                                                        config_.similarity_threshold);

    return build_rag_prompt(query, docs);
}

std::string RAGRetriever::build_rag_prompt(const std::string& query,
                                           const std::vector<SearchResult>& docs) const
{
    std::stringstream context;

    for (size_t i = 0; i < docs.size(); ++i) {
        context << "[Doc " << (i + 1) << "]\n";
        context << docs[i].text << "\n\n";
    }

    std::stringstream prompt;
    prompt << "[SYSTEM] " << config_.system_prompt << "\n\n";
    prompt << "[DOCUMENTI]\n" << context.str() << "\n";
    prompt << "[DOMANDA] " << query << "\n";
    prompt << "[RISPOSTA]";

    return prompt.str();
}

void RAGRetriever::clear()
{
    vector_db_.clear();
}

void RAGRetriever::set_embedder(EmbedFn embedder)
{
    embedder_ = std::move(embedder);
}

std::vector<float> RAGRetriever::embed(const std::string& text) const
{
    if (embedder_) {
        std::vector<float> v = embedder_(text);
        // The vector DB requires exactly config_.embedding_dim components;
        // a mismatched backend would silently corrupt the index, so fall
        // back to the hash embedder instead of resizing.
        if (v.size() == config_.embedding_dim) {
            normalize_embedding(v);
            return v;
        }
    }
    return compute_embedding(text, config_.embedding_dim);
}

}  // namespace rag
}  // namespace serius
