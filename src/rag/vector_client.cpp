#include "vector_client.h"

#include <algorithm>
#include <cmath>

namespace serius {
namespace rag {

VectorDBClient::VectorDBClient(const Config& config)
    : config_(config)
{
}

VectorDBClient::~VectorDBClient()
{
}

float VectorDBClient::cosine_similarity(const std::vector<float>& a,
                                       const std::vector<float>& b) const
{
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (norm_a < 1e-6f || norm_b < 1e-6f) {
        return 0.0f;
    }

    return dot / (norm_a * norm_b);
}

std::vector<SearchResult> VectorDBClient::search(
    const std::vector<float>& query_embedding,
    int top_k, float threshold)
{
    std::vector<SearchResult> results;

    // Calculate similarity for all documents
    std::vector<std::pair<size_t, float>> similarities;
    for (size_t i = 0; i < documents_.size(); ++i) {
        if (documents_[i].embedding.size() == query_embedding.size()) {
            float sim = cosine_similarity(query_embedding, documents_[i].embedding);
            if (sim >= threshold) {
                similarities.emplace_back(i, sim);
            }
        }
    }

    // Sort by similarity (descending)
    std::sort(similarities.begin(), similarities.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Take top-k
    int count = std::min(top_k, static_cast<int>(similarities.size()));
    for (int i = 0; i < count; ++i) {
        size_t idx = similarities[i].first;
        SearchResult result;
        result.id = documents_[idx].id;
        result.text = documents_[idx].text;
        result.score = similarities[i].second;
        result.metadata = documents_[idx].metadata;
        results.push_back(result);
    }

    return results;
}

void VectorDBClient::upsert(const std::string& id,
                             const std::vector<float>& embedding,
                             const std::string& text,
                             const json& metadata)
{
    // Check if document already exists
    auto it = std::find_if(documents_.begin(), documents_.end(),
                          [&id](const Document& doc) { return doc.id == id; });

    if (it != documents_.end()) {
        // Update existing
        it->embedding = embedding;
        it->text = text;
        it->metadata = metadata;
    } else {
        // Insert new
        Document doc;
        doc.id = id;
        doc.embedding = embedding;
        doc.text = text;
        doc.metadata = metadata;
        documents_.push_back(std::move(doc));
    }
}

void VectorDBClient::remove(const std::string& id)
{
    documents_.erase(
        std::remove_if(documents_.begin(), documents_.end(),
                      [&id](const Document& doc) { return doc.id == id; }),
        documents_.end());
}

SearchResult VectorDBClient::get(const std::string& id) const
{
    auto it = std::find_if(documents_.begin(), documents_.end(),
                          [&id](const Document& doc) { return doc.id == id; });

    if (it != documents_.end()) {
        SearchResult result;
        result.id = it->id;
        result.text = it->text;
        result.score = 1.0f;  // Exact match
        result.metadata = it->metadata;
        return result;
    }

    return SearchResult{};
}

void VectorDBClient::clear()
{
    documents_.clear();
}

}  // namespace rag
}  // namespace serius
