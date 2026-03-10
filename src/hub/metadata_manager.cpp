#include "metadata_manager.h"
#include <sstream>
#include <spdlog/spdlog.h>

namespace serius::hub {

MetadataManager::MetadataManager(const std::string& db_path) : db_path_(db_path), db_(nullptr) {}

MetadataManager::~MetadataManager() {
    if (db_) {
        sqlite3_close(db_);
    }
}

bool MetadataManager::initialize() {
    if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK) {
        spdlog::error("Failed to open database: {}", sqlite3_errmsg(db_));
        return false;
    }
    
    return create_tables();
}

bool MetadataManager::create_tables() {
    const char* create_models_table = R"(
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            author TEXT NOT NULL,
            version TEXT NOT NULL,
            license TEXT,
            tags TEXT,  -- JSON array
            model_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            checksum_sha256 TEXT NOT NULL,
            config TEXT,  -- JSON object
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            download_count INTEGER DEFAULT 0,
            rating REAL DEFAULT 0.0,
            is_public BOOLEAN DEFAULT 1
        )
    )";

    const char* create_fts_table = R"(
        CREATE VIRTUAL TABLE IF NOT EXISTS models_fts USING fts5(
            id, name, description, author, tags,
            content='models',
            content_rowid='rowid'
        )
    )";

    const char* create_triggers = R"(
        CREATE TRIGGER IF NOT EXISTS models_ai AFTER INSERT ON models BEGIN
            INSERT INTO models_fts(id, name, description, author, tags)
            VALUES (new.id, new.name, new.description, new.author, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS models_ad AFTER DELETE ON models BEGIN
            INSERT INTO models_fts(models_fts, id, name, description, author, tags)
            VALUES ('delete', old.id, old.name, old.description, old.author, old.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS models_au AFTER UPDATE ON models BEGIN
            INSERT INTO models_fts(models_fts, id, name, description, author, tags)
            VALUES ('delete', old.id, old.name, old.description, old.author, old.tags);
            INSERT INTO models_fts(id, name, description, author, tags)
            VALUES (new.id, new.name, new.description, new.author, new.tags);
        END;
    )";

    return execute_query(create_models_table) && 
           execute_query(create_fts_table) && 
           execute_query(create_triggers);
}

bool MetadataManager::save_model_metadata(const ModelMetadata& metadata) {
    const char* sql = R"(
        INSERT INTO models (id, name, description, author, version, license, tags,
                          model_type, file_size, checksum_sha256, config,
                          created_at, updated_at, download_count, rating, is_public)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_text(stmt, 1, metadata.id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, metadata.name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, metadata.description.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, metadata.author.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 5, metadata.version.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 6, metadata.license.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 7, nlohmann::json(metadata.tags).dump().c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 8, metadata.model_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 9, static_cast<int>(metadata.file_size));
    sqlite3_bind_text(stmt, 10, metadata.checksum_sha256.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 11, metadata.config.dump().c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 12, metadata.created_at.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 13, metadata.updated_at.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 14, metadata.download_count);
    sqlite3_bind_double(stmt, 15, metadata.rating);
    sqlite3_bind_int(stmt, 16, metadata.is_public ? 1 : 0);

    bool result = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    return result;
}

std::vector<ModelMetadata> MetadataManager::search_models(
    const std::string& query,
    const std::vector<std::string>& tags,
    const std::string& author,
    int limit, int offset) {
    
    std::string sql = R"(
        SELECT m.* FROM models m
        LEFT JOIN models_fts fts ON m.id = fts.id
        WHERE m.is_public = 1
    )";
    
    std::vector<std::string> params;
    
    if (!query.empty()) {
        sql += " AND models_fts MATCH ?";
        params.push_back(query);
    }
    
    if (!author.empty()) {
        sql += " AND m.author = ?";
        params.push_back(author);
    }
    
    if (!tags.empty()) {
        sql += " AND (";
        for (size_t i = 0; i < tags.size(); i++) {
            sql += "m.tags LIKE ?";
            params.push_back("%" + tags[i] + "%");
            if (i < tags.size() - 1) sql += " OR ";
        }
        sql += ")";
    }
    
    sql += " ORDER BY m.download_count DESC, m.rating DESC LIMIT ? OFFSET ?";
    params.push_back(std::to_string(limit));
    params.push_back(std::to_string(offset));
    
    return execute_search_query(sql, params);
}

} // namespace serius::hub
