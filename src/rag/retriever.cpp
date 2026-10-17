#include "retriever.h"

#include <sstream>

namespace serius {
namespace rag {

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
    std::vector<float> embedding = compute_embedding(text, config_.embedding_dim);
    vector_db_.upsert(id, embedding, text, metadata);
}

std::string RAGRetriever::retrieve(const std::string& query)
{
    std::vector<float> query_embedding = compute_embedding(query, config_.embedding_dim);
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

}  // namespace rag
}  // namespace serius
