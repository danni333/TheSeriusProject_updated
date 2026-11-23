#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
// #include <sqlite3.h>

#include "../common/text_embed.h"
#include "../tensor/tensor.h"
#include "../tokenizer/bpe.h"
#include "nomic_bert.h"

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

    // Task-aware variants: nomic-embed models are trained with
    // "search_query: "/"search_document: " prefixes — prepending them
    // measurably improves retrieval. Other backends get the raw text.
    std::vector<float> compute_query_embedding(const std::string &text);
    std::vector<float> compute_document_embedding(const std::string &text);

    // Batch compute embeddings
    std::vector<std::vector<float>> compute_embeddings(const std::vector<std::string> &texts);

    /// Embedder factories: return `common::TextEmbedFn` callables bound to
    /// this manager, ready to plug into scorers that accept an embedder
    /// (grounding::SupportScorer, safety::ToxicityClassifier,
    /// reasoning::EmbeddingVerifier). The callables hold a pointer to `this`:
    /// the EmbeddingManager must outlive them.
    common::TextEmbedFn query_embed_fn();
    common::TextEmbedFn document_embed_fn();

    // Model management
    bool load_model(const std::string &model_path);
    void unload_model();
    bool is_model_loaded() const;

   private:
    void *model_handle_;
    std::string model_path_;
    bool model_loaded_;

    // Real embedding path: token embedding matrix [vocab, dim] mean-pooled
    // over BPE token ids, L2-normalized. Empty matrix -> hash fallback.
    core::Tensor embed_matrix_;
    std::unique_ptr<serius::BPETokenizer> tokenizer_;

    // Full encoder path for embedding-specialized GGUFs (nomic-bert arch):
    // WordPiece + transformer forward + mean pooling — real sentence
    // embeddings instead of token-embedding lookup.
    std::unique_ptr<NomicBertEmbedder> bert_;
    bool use_task_prefixes_ = false;  // nomic-bert: prepend search_query/document

    bool try_load_gguf(const std::string &path);
    bool try_load_nomic_gguf(const std::string &path);
    bool try_load_safetensors_dir(const std::string &dir);

    // Embedding computation (deterministic hashed bag-of-words fallback when
    // no embedding model file is provided to load_model()).
    std::vector<float> compute_with_simple_tokenizer(const std::string &text);
};

}  // namespace serius::hub
