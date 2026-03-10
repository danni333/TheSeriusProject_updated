#include "search_engine.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <spdlog/spdlog.h>

namespace serius::hub {

SearchEngine::SearchEngine(const std::string& db_path) 
    : db_path_(db_path), db_(nullptr), embedding_manager_(std::make_unique<EmbeddingManager>()) {}

SearchEngine::~SearchEngine() {
    if (db_) {
        sqlite3_close(db_);
    }
}

bool SearchEngine::initialize() {
    if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK) {
        spdlog::error("Failed to open search database: {}", sqlite3_errmsg(db_));
        return false;
    }
    
    if (!create_tables()) {
        return false;
    }
    
    // Initialize embedding manager
    return embedding_manager_->initialize("models/embedding_model.bin");
}

bool SearchEngine::create_tables() {
    const char* create_search_table = R"(
        CREATE TABLE IF NOT EXISTS search_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            user_id TEXT,
            results_count INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    )";

    const char* create_embeddings_table = R"(
        CREATE TABLE IF NOT EXISTS model_embeddings (
            model_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models(id)
        )
    )";

    const char* create_trending_table = R"(
        CREATE TABLE IF NOT EXISTS trending_searches (
            query TEXT PRIMARY KEY,
            search_count INTEGER DEFAULT 1,
            last_searched DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    )";

    return execute_query(create_search_table) && 
           execute_query(create_embeddings_table) && 
           execute_query(create_trending_table);
}

std::vector<SearchResult> SearchEngine::search_models(const SearchFilter& filter, 
                                                    int limit, int offset) {
    // Check cache first
    std::string cache_key = filter.query + "_" + std::to_string(limit) + "_" + std::to_string(offset);
    if (is_cache_valid(cache_key)) {
        return get_cached_results(cache_key);
    }
    
    std::vector<std::vector<SearchResult>> result_sets;
    
    // Full-text search
    if (!filter.query.empty()) {
        auto ft_results = fulltext_search(filter, limit, offset);
        result_sets.push_back(ft_results);
    }
    
    // Tag search
    if (!filter.tags.empty()) {
        auto tag_results = tag_search(filter.tags, limit, offset);
        result_sets.push_back(tag_results);
    }
    
    // Author search
    if (!filter.author.empty()) {
        auto author_results = author_search(filter.author, limit, offset);
        result_sets.push_back(author_results);
    }
    
    // Semantic search if embedding model is available
    if (embedding_manager_->is_model_loaded() && !filter.query.empty()) {
        auto semantic_results = semantic_search(filter.query, filter.tags, limit, offset);
        result_sets.push_back(semantic_results);
    }
    
    // Merge and rank results
    auto results = merge_and_rank_results(result_sets);
    
    // Apply additional filters
    results.erase(std::remove_if(results.begin(), results.end(),
        [&filter](const SearchResult& result) {
            if (filter.min_rating > 0.0f) {
                // Would need to fetch rating from database
            }
            if (filter.min_downloads > 0) {
                // Would need to fetch download count from database
            }
            return false;
        }), results.end());
    
    // Cache results
    cache_results(cache_key, results);
    
    return results;
}

std::vector<SearchResult> SearchEngine::fulltext_search(const SearchFilter& filter,
                                                       int limit, int offset) {
    std::string sql = R"(
        SELECT m.id, m.name, m.description, m.author, 
               bm25(models_fts) as score
        FROM models m
        JOIN models_fts ON m.id = models_fts.id
        WHERE m.is_public = 1
    )";
    
    std::vector<std::string> params;
    
    if (!filter.query.empty()) {
        sql += " AND models_fts MATCH ?";
        params.push_back(filter.query);
    }
    
    if (!filter.model_type.empty()) {
        sql += " AND m.model_type = ?";
        params.push_back(filter.model_type);
    }
    
    if (!filter.license.empty()) {
        sql += " AND m.license = ?";
        params.push_back(filter.license);
    }
    
    sql += " ORDER BY score DESC LIMIT ? OFFSET ?";
    params.push_back(std::to_string(limit));
    params.push_back(std::to_string(offset));
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return {};
    }
    
    // Bind parameters
    for (size_t i = 0; i < params.size(); i++) {
        sqlite3_bind_text(stmt, i + 1, params[i].c_str(), -1, SQLITE_STATIC);
    }
    
    std::vector<SearchResult> results;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        float score = static_cast<float>(sqlite3_column_double(stmt, 4));
        SearchResult result = row_to_search_result(stmt, score);
        result.highlighted_text = highlight_text(
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)),
            filter.query);
        results.push_back(result);
    }
    
    sqlite3_finalize(stmt);
    return results;
}

std::vector<SearchResult> SearchEngine::semantic_search(const std::string& query,
                                                       const std::vector<std::string>& tags,
                                                       int limit, int offset) {
    // Compute query embedding
    auto query_embedding = compute_query_embedding(query);
    if (query_embedding.empty()) {
        return {};
    }
    
    // Fetch all model embeddings from database
    std::string sql = R"(
        SELECT m.id, m.name, m.description, m.author, e.embedding
        FROM models m
        JOIN model_embeddings e ON m.id = e.model_id
        WHERE m.is_public = 1
        LIMIT 1000
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return {};
    }
    
    std::vector<std::pair<SearchResult, std::vector<float>>> model_embeddings;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        SearchResult result = row_to_search_result(stmt, 0.0f);
        
        // Extract embedding from BLOB
        const void* blob = sqlite3_column_blob(stmt, 4);
        int blob_size = sqlite3_column_bytes(stmt, 4);
        std::vector<float> embedding(blob_size / sizeof(float));
        std::memcpy(embedding.data(), blob, blob_size);
        
        model_embeddings.emplace_back(result, embedding);
    }
    
    sqlite3_finalize(stmt);
    
    // Compute similarities
    std::vector<std::pair<float, SearchResult>> similarities;
    for (auto& [result, embedding] : model_embeddings) {
        float similarity = cosine_similarity(query_embedding, embedding);
        similarities.emplace_back(similarity, result);
    }
    
    // Sort by similarity
    std::sort(similarities.rbegin(), similarities.rend());
    
    // Convert to results
    std::vector<SearchResult> results;
    for (size_t i = offset; i < std::min(similarities.size(), offset + limit); i++) {
        auto [similarity, result] = similarities[i];
        result.score = similarity;
        results.push_back(result);
    }
    
    return results;
}

std::vector<float> SearchEngine::compute_query_embedding(const std::string& query) {
    return embedding_manager_->compute_embedding(query);
}

float SearchEngine::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); i++) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }
    
    return dot_product / (norm_a * norm_b);
}

SearchResult SearchEngine::row_to_search_result(sqlite3_stmt* stmt, float score) {
    SearchResult result;
    result.model_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    result.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    result.description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    result.author = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    result.score = score;
    return result;
}

std::string SearchEngine::highlight_text(const std::string& text, const std::string& query) {
    // Simple highlighting implementation
    // In production, this would use more sophisticated highlighting
    std::string highlighted = text;
    size_t pos = highlighted.find(query);
    if (pos != std::string::npos) {
        highlighted.replace(pos, query.length(), "**" + query + "**");
    }
    return highlighted;
}

bool SearchEngine::is_cache_valid(const std::string& cache_key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = search_cache_.find(cache_key);
    return it != search_cache_.end();
}

void SearchEngine::cache_results(const std::string& cache_key, const std::vector<SearchResult>& results) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    search_cache_[cache_key] = results;
}

std::vector<SearchResult> SearchEngine::get_cached_results(const std::string& cache_key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = search_cache_.find(cache_key);
    return it != search_cache_.end() ? it->second : std::vector<SearchResult>();
}

// EmbeddingManager implementation
EmbeddingManager::EmbeddingManager() : model_handle_(nullptr), model_loaded_(false) {}

EmbeddingManager::~EmbeddingManager() {
    unload_model();
}

bool EmbeddingManager::initialize(const std::string& model_path) {
    return load_model(model_path);
}

std::vector<float> EmbeddingManager::compute_embedding(const std::string& text) {
    if (!model_loaded_) {
        return {};
    }
    
    // For now, return a simple hash-based embedding
    // In production, this would use a proper embedding model
    std::vector<float> embedding(384, 0.0f);  // Standard embedding size
    
    for (size_t i = 0; i < text.size() && i < embedding.size(); i++) {
        embedding[i] = static_cast<float>(text[i]) / 255.0f;
    }
    
    // Normalize
    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 0.0f) {
        for (float& val : embedding) {
            val /= norm;
        }
    }
    
    return embedding;
}

bool EmbeddingManager::load_model(const std::string& model_path) {
    // Simplified model loading
    // In production, this would load a real embedding model
    model_path_ = model_path;
    model_loaded_ = true;
    return true;
}

void EmbeddingManager::unload_model() {
    model_loaded_ = false;
    model_path_.clear();
}

bool EmbeddingManager::is_model_loaded() const {
    return model_loaded_;
}

} // namespace serius::hub
