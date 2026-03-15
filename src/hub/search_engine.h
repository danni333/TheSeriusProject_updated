#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
// #include <sqlite3.h>

namespace serius::hub {

struct SearchFilter
{
    std::string query;
    std::vector<std::string> tags;
    std::string author;
    std::string model_type;
    std::string license;
    float min_rating = 0.0f;
    int min_downloads = 0;
    std::string date_after;
    std::string date_before;
    bool sort_by_popularity = false;
    bool sort_by_rating = false;
    bool sort_by_date = false;
};

struct SearchResult
{
    std::string model_id;
    std::string name;
    std::string description;
    std::string author;
    float score;
    std::vector<std::string> matched_tags;
    std::string highlighted_text;
};

class SearchEngine
{
   public:
    SearchEngine(const std::string &db_path);
    ~SearchEngine();

    bool initialize();

    // Text search
    std::vector<SearchResult> search_models(const SearchFilter &filter, int limit = 50, int offset = 0);

    // Semantic search (using embeddings)
    std::vector<SearchResult> semantic_search(const std::string &query, const std::vector<std::string> &tags = {},
                                              int limit = 50, int offset = 0);

    // Similarity search
    std::vector<SearchResult> find_similar_models(const std::string &model_id, int limit = 10);

    // Auto-complete suggestions
    std::vector<std::string> get_suggestions(const std::string &partial_query, int limit = 10);

    // Popular searches
    std::vector<std::string> get_trending_searches(int limit = 10);

    // Search analytics
    void log_search(const std::string &query, const std::string &user_id, int results_count);
    std::vector<std::string> get_popular_queries(int limit = 100);

   private:
    // sqlite3 *db_;
    std::string db_path_;

    // Embedding-based search
    std::unique_ptr<class EmbeddingManager> embedding_manager_;

    bool create_tables();
    bool execute_query(const std::string &sql);

    // Text search implementations
    std::vector<SearchResult> fulltext_search(const SearchFilter &filter, int limit, int offset);
    std::vector<SearchResult> tag_search(const std::vector<std::string> &tags, int limit, int offset);
    std::vector<SearchResult> author_search(const std::string &author, int limit, int offset);

    // Semantic search implementations
    std::vector<float> compute_query_embedding(const std::string &query);
    float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b);

    // Result processing
    std::vector<SearchResult> merge_and_rank_results(const std::vector<std::vector<SearchResult>> &result_sets);
    // SearchResult row_to_search_result(sqlite3_stmt *stmt, float score = 0.0f);
    std::string highlight_text(const std::string &text, const std::string &query);

    // Caching
    std::map<std::string, std::vector<SearchResult>> search_cache_;
    std::mutex cache_mutex_;
    std::chrono::seconds cache_ttl_{300};  // 5 minutes

    bool is_cache_valid(const std::string &cache_key);
    void cache_results(const std::string &cache_key, const std::vector<SearchResult> &results);
    std::vector<SearchResult> get_cached_results(const std::string &cache_key);
};

// Embedding manager for semantic search
class EmbeddingManager
{
   public:
    EmbeddingManager();
    ~EmbeddingManager();

    bool initialize(const std::string &model_path);

    // Compute embeddings for text
    std::vector<float> compute_embedding(const std::string &text);

    // Batch compute embeddings
    std::vector<std::vector<float>> compute_embeddings(const std::vector<std::string> &texts);

    // Model management
    bool load_model(const std::string &model_path);
    void unload_model();
    bool is_model_loaded() const;

   private:
    void *model_handle_;
    std::string model_path_;
    bool model_loaded_;

    // Embedding computation
    std::vector<float> compute_with_sentence_transformer(const std::string &text);
    std::vector<float> compute_with_simple_tokenizer(const std::string &text);
};

}  // namespace serius::hub
