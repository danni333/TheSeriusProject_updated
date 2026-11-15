#include "metadata_manager.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef SERIUS_HUB_SQLITE
#include <sqlite3.h>
#endif


namespace serius::hub {

MetadataManager::MetadataManager(const std::string &db_path) : db_path_(db_path) {}

MetadataManager::~MetadataManager()
{
#ifdef SERIUS_HUB_SQLITE
    if (db_) {
        sqlite3_close(db_);
    }
#endif
}

bool MetadataManager::initialize()
{
#ifdef SERIUS_HUB_SQLITE
    return ensure_open();
#else
    return true;
#endif
}

bool MetadataManager::save_model_metadata(const ModelMetadata &metadata)
{
    auto stored = metadata;
    if (stored.created_at.empty()) stored.created_at = now_iso8601();
    stored.updated_at = now_iso8601();

#ifdef SERIUS_HUB_SQLITE
    if (!use_json_fallback_ && ensure_open()) {
        const char *sql =
            "INSERT OR REPLACE INTO models (id, name, description, author, version, license,"
            " tags, model_type, file_size, checksum_sha256, config, created_at, updated_at,"
            " download_count, rating, is_public)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            spdlog::error("MetadataManager: prepare failed: {}", sqlite3_errmsg(db_));
            return false;
        }
        const bool ok = bind_and_step_insert(stmt, stored);
        sqlite3_finalize(stmt);
        return ok;
    }
    if (!use_json_fallback_) return false;
#endif

    auto db = load_db();
    db[stored.id] = model_to_json(stored);
    return save_db(db);
}

bool MetadataManager::update_model_metadata(const std::string &model_id, const ModelMetadata &metadata)
{
    auto updated = metadata;
    updated.id = model_id;
    updated.updated_at = now_iso8601();

#ifdef SERIUS_HUB_SQLITE
    if (!use_json_fallback_ && ensure_open()) {
        const char *sql =
            "UPDATE models SET name=?, description=?, author=?, version=?, license=?,"
            " tags=?, model_type=?, file_size=?, checksum_sha256=?, config=?,"
            " updated_at=?, download_count=?, rating=?, is_public=? WHERE id=?";
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            spdlog::error("MetadataManager: prepare failed: {}", sqlite3_errmsg(db_));
            return false;
        }
        sqlite3_bind_text(stmt, 1, updated.name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, updated.description.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, updated.author.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, updated.version.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, updated.license.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, nlohmann::json(updated.tags).dump().c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 7, updated.model_type.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 8, static_cast<sqlite3_int64>(updated.file_size));
        sqlite3_bind_text(stmt, 9, updated.checksum_sha256.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 10, updated.config.dump().c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 11, updated.updated_at.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 12, updated.download_count);
        sqlite3_bind_double(stmt, 13, updated.rating);
        sqlite3_bind_int(stmt, 14, updated.is_public ? 1 : 0);
        sqlite3_bind_text(stmt, 15, model_id.c_str(), -1, SQLITE_TRANSIENT);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE && sqlite3_changes(db_) > 0;
        sqlite3_finalize(stmt);
        return ok;
    }
    if (!use_json_fallback_) return false;
#endif

    auto db = load_db();
    if (!db.is_object()) db = nlohmann::json::object();
    if (!db.contains(model_id)) return false;
    db[model_id] = model_to_json(updated);
    return save_db(db);
}

ModelMetadata MetadataManager::get_model_metadata(const std::string &model_id)
{
#ifdef SERIUS_HUB_SQLITE
    if (!use_json_fallback_ && ensure_open()) {
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, "SELECT * FROM models WHERE id = ?", -1, &stmt, nullptr) != SQLITE_OK) {
            return ModelMetadata{};
        }
        sqlite3_bind_text(stmt, 1, model_id.c_str(), -1, SQLITE_TRANSIENT);
        ModelMetadata m;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            m = row_to_model(stmt);
        }
        sqlite3_finalize(stmt);
        return m;
    }
    if (!use_json_fallback_) return ModelMetadata{};
#endif
    const auto db = load_db();
    if (db.is_object() && db.contains(model_id)) {
        return json_to_model(db[model_id]);
    }
    return ModelMetadata{};
}

std::vector<ModelMetadata> MetadataManager::list_all_models()
{
#ifdef SERIUS_HUB_SQLITE
    if (!use_json_fallback_ && ensure_open()) {
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, "SELECT * FROM models", -1, &stmt, nullptr) != SQLITE_OK) {
            return {};
        }
        std::vector<ModelMetadata> out;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            out.push_back(row_to_model(stmt));
        }
        sqlite3_finalize(stmt);
        return out;
    }
    if (!use_json_fallback_) return {};
#endif
    const auto db = load_db();
    std::vector<ModelMetadata> out;
    if (db.is_object()) {
        for (auto &[id, j] : db.items()) {
            (void)id;
            out.push_back(json_to_model(j));
        }
    }
    return out;
}

namespace {

/// Escape LIKE wildcards so user input matches literally.
std::string escape_like(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '%' || c == '_' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

}  // namespace

std::vector<ModelMetadata> MetadataManager::search_models(const std::string &query,
                                                          const std::vector<std::string> &tags,
                                                          const std::string &author, int limit, int offset)
{
#ifdef SERIUS_HUB_SQLITE
    if (!use_json_fallback_ && ensure_open()) {

        std::string sql = "SELECT * FROM models WHERE 1=1";
        std::vector<std::string> params;
        // SQLite LIKE is case-insensitive for ASCII — no lowering needed.
        if (!query.empty()) {
            sql += " AND (name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\')";
            const std::string pat = "%" + escape_like(query) + "%";
            params.push_back(pat);
            params.push_back(pat);
        }
        if (!author.empty()) {
            sql += " AND author = ?";
            params.push_back(author);
        }
        // tags are stored as a JSON array text; the JSON fallback matched ANY
        // of the requested tags, so group the LIKEs with OR.
        if (!tags.empty()) {
            sql += " AND (";
            for (size_t i = 0; i < tags.size(); ++i) {
                if (i) sql += " OR ";
                sql += "tags LIKE ? ESCAPE '\\'";
                params.push_back("%\"" + escape_like(tags[i]) + "\"%");
            }
            sql += ")";
        }
        sql += " ORDER BY download_count DESC LIMIT ? OFFSET ?";

        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            spdlog::error("MetadataManager: search prepare failed: {}", sqlite3_errmsg(db_));
            return {};
        }
        for (size_t i = 0; i < params.size(); ++i) {
            sqlite3_bind_text(stmt, static_cast<int>(i + 1), params[i].c_str(), -1, SQLITE_TRANSIENT);
        }
        const int lim_idx = static_cast<int>(params.size() + 1);
        sqlite3_bind_int(stmt, lim_idx, limit);
        sqlite3_bind_int(stmt, lim_idx + 1, offset);
        std::vector<ModelMetadata> results;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            results.push_back(row_to_model(stmt));
        }
        sqlite3_finalize(stmt);
        return results;
    }
    if (!use_json_fallback_) return {};
#endif

    std::vector<ModelMetadata> results;

    const auto db = load_db();
    if (!db.is_object()) return results;

    for (auto &[id, model_json] : db.items()) {
        ModelMetadata metadata = json_to_model(model_json);

        // Apply filters
        if (!query.empty()) {
            std::string search_text = metadata.name + " " + metadata.description;
            std::transform(search_text.begin(), search_text.end(), search_text.begin(), ::tolower);
            std::string query_lower = query;
            std::transform(query_lower.begin(), query_lower.end(), query_lower.begin(), ::tolower);

            if (search_text.find(query_lower) == std::string::npos) {
                continue;
            }
        }

        if (!author.empty() && metadata.author != author) {
            continue;
        }

        if (!tags.empty()) {
            bool has_tag = false;
            for (const auto &tag : tags) {
                if (std::find(metadata.tags.begin(), metadata.tags.end(), tag) != metadata.tags.end()) {
                    has_tag = true;
                    break;
                }
            }
            if (!has_tag) continue;
        }

        results.push_back(metadata);
    }

    // Sort by download count
    std::sort(results.begin(), results.end(),
              [](const ModelMetadata &a, const ModelMetadata &b) { return a.download_count > b.download_count; });

    // Apply pagination
    if (offset >= results.size()) return {};
    const size_t begin = static_cast<size_t>(offset);
    const size_t end = std::min(results.size(), begin + static_cast<size_t>(limit));
    results.erase(results.begin() + end, results.end());
    results.erase(results.begin(), results.begin() + begin);

    return results;
}

bool MetadataManager::delete_model_metadata(const std::string &model_id)
{
#ifdef SERIUS_HUB_SQLITE
    if (!use_json_fallback_ && ensure_open()) {
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, "DELETE FROM models WHERE id = ?", -1, &stmt, nullptr) != SQLITE_OK) {
            return false;
        }
        sqlite3_bind_text(stmt, 1, model_id.c_str(), -1, SQLITE_TRANSIENT);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE && sqlite3_changes(db_) > 0;
        sqlite3_finalize(stmt);
        return ok;
    }
#endif
    auto db = load_db();
    if (!db.is_object() || !db.contains(model_id)) return false;
    db.erase(model_id);
    return save_db(db);
}

int MetadataManager::get_model_count()
{
#ifdef SERIUS_HUB_SQLITE
    if (!use_json_fallback_ && ensure_open()) {
        sqlite3_stmt *stmt = nullptr;
        int count = 0;
        if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM models", -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
        return count;
    }
    if (!use_json_fallback_) return 0;
#endif
    const auto db = load_db();
    return db.is_object() ? static_cast<int>(db.size()) : 0;
}

int MetadataManager::get_download_count(const std::string &model_id)
{
    return get_model_metadata(model_id).download_count;
}

bool MetadataManager::increment_download_count(const std::string &model_id)
{
#ifdef SERIUS_HUB_SQLITE
    if (!use_json_fallback_ && ensure_open()) {
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, "UPDATE models SET download_count = download_count + 1 WHERE id = ?",
                               -1, &stmt, nullptr) != SQLITE_OK) {
            return false;
        }
        sqlite3_bind_text(stmt, 1, model_id.c_str(), -1, SQLITE_TRANSIENT);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE && sqlite3_changes(db_) > 0;
        sqlite3_finalize(stmt);
        return ok;
    }
#endif
    auto db = load_db();
    if (!db.is_object() || !db.contains(model_id)) return false;
    auto m = json_to_model(db[model_id]);
    m.download_count += 1;
    db[model_id] = model_to_json(m);
    return save_db(db);
}

bool MetadataManager::update_rating(const std::string &model_id, float rating)
{
#ifdef SERIUS_HUB_SQLITE
    if (!use_json_fallback_ && ensure_open()) {
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, "UPDATE models SET rating = ? WHERE id = ?", -1, &stmt, nullptr) !=
            SQLITE_OK) {
            return false;
        }
        sqlite3_bind_double(stmt, 1, rating);
        sqlite3_bind_text(stmt, 2, model_id.c_str(), -1, SQLITE_TRANSIENT);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE && sqlite3_changes(db_) > 0;
        sqlite3_finalize(stmt);
        return ok;
    }
#endif
    auto db = load_db();
    if (!db.is_object() || !db.contains(model_id)) return false;
    auto m = json_to_model(db[model_id]);
    m.rating = rating;
    db[model_id] = model_to_json(m);
    return save_db(db);
}

nlohmann::json MetadataManager::load_db() const
{
    std::ifstream file(db_path_);
    if (!file.is_open()) return nlohmann::json::object();
    try {
        nlohmann::json j;
        file >> j;
        return j;
    } catch (const std::exception &e) {
        std::cerr << "[Hub] Corrupt metadata DB " << db_path_ << ": " << e.what() << "\n";
        return nlohmann::json::object();
    }
}

bool MetadataManager::save_db(const nlohmann::json &db)
{
    std::ofstream out(db_path_);
    if (!out.is_open()) return false;
    out << db.dump(2);
    return true;
}

#ifdef SERIUS_HUB_SQLITE

bool MetadataManager::create_tables()
{
    const char *sql =
        "CREATE TABLE IF NOT EXISTS models ("
        " id TEXT PRIMARY KEY,"
        " name TEXT NOT NULL DEFAULT '',"
        " description TEXT NOT NULL DEFAULT '',"
        " author TEXT NOT NULL DEFAULT '',"
        " version TEXT NOT NULL DEFAULT '',"
        " license TEXT NOT NULL DEFAULT '',"
        " tags TEXT NOT NULL DEFAULT '[]',"
        " model_type TEXT NOT NULL DEFAULT '',"
        " file_size INTEGER NOT NULL DEFAULT 0,"
        " checksum_sha256 TEXT NOT NULL DEFAULT '',"
        " config TEXT NOT NULL DEFAULT '{}',"
        " created_at TEXT NOT NULL DEFAULT '',"
        " updated_at TEXT NOT NULL DEFAULT '',"
        " download_count INTEGER NOT NULL DEFAULT 0,"
        " rating REAL NOT NULL DEFAULT 0,"
        " is_public INTEGER NOT NULL DEFAULT 1"
        ")";
    char *err = nullptr;
    if (sqlite3_exec(db_, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        spdlog::error("MetadataManager: create_tables failed: {}", err ? err : "?");
        sqlite3_free(err);
        return false;
    }
    return true;
}

bool MetadataManager::execute_query(const std::string &sql)
{
    char *err = nullptr;
    if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
        spdlog::error("MetadataManager: query failed: {}", err ? err : "?");
        sqlite3_free(err);
        return false;
    }
    return true;
}

bool MetadataManager::ensure_open()
{
    if (db_ || use_json_fallback_) return true;

    // One-time migration: an existing non-SQLite file at db_path_ is treated
    // as legacy JSON metadata, imported into the fresh database, then renamed
    // aside so it is never parsed as JSON again.
    std::vector<ModelMetadata> pending_import;
    if (db_path_ != ":memory:" && std::filesystem::exists(db_path_)) {
        bool is_sqlite = false;
        {
            // Scope the probe so the handle is closed before we rename the
            // file — Windows refuses to rename an open file.
            std::ifstream probe(db_path_, std::ios::binary);
            char magic[16] = {};
            probe.read(magic, sizeof(magic));
            is_sqlite = probe.gcount() == 16 &&
                        std::memcmp(magic, "SQLite format 3\0", 16) == 0;
        }
        if (!is_sqlite) {
            const auto legacy = load_db();
            if (legacy.is_object()) {
                for (auto &[id, j] : legacy.items()) {
                    (void)id;
                    ModelMetadata m = json_to_model(j);
                    if (m.id.empty()) m.id = id;
                    if (!m.id.empty()) pending_import.push_back(std::move(m));
                }
            }
            const std::string bak = db_path_ + ".migrated.json";
            std::error_code ec;
            std::filesystem::rename(db_path_, bak, ec);
            if (ec) {
                spdlog::warn("MetadataManager: could not rename legacy JSON {}: {}; "
                             "keeping JSON backend", db_path_, ec.message());
                pending_import.clear();
                use_json_fallback_ = true;
                return true;
            }
            spdlog::info("MetadataManager: migrated {} model(s) from JSON {} -> {}",
                         pending_import.size(), db_path_, bak);
        }
    }

    if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK) {
        spdlog::error("MetadataManager: cannot open db '{}': {}", db_path_,
                      db_ ? sqlite3_errmsg(db_) : "out of memory");
        if (db_) { sqlite3_close(db_); db_ = nullptr; }
        return false;
    }
    sqlite3_busy_timeout(db_, 5000);
    if (!create_tables()) return false;

    for (const auto &m : pending_import) {
        const char *sql =
            "INSERT OR REPLACE INTO models (id, name, description, author, version, license,"
            " tags, model_type, file_size, checksum_sha256, config, created_at, updated_at,"
            " download_count, rating, is_public)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        sqlite3_stmt *stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            bind_and_step_insert(stmt, m);
            sqlite3_finalize(stmt);
        }
    }
    return true;
}

ModelMetadata MetadataManager::row_to_model(sqlite3_stmt *stmt)
{
    ModelMetadata m;
    auto col_text = [&](int i) -> std::string {
        const unsigned char *s = sqlite3_column_text(stmt, i);
        return s ? reinterpret_cast<const char *>(s) : "";
    };
    m.id = col_text(0);
    m.name = col_text(1);
    m.description = col_text(2);
    m.author = col_text(3);
    m.version = col_text(4);
    m.license = col_text(5);
    try {
        m.tags = nlohmann::json::parse(col_text(6)).get<std::vector<std::string>>();
    } catch (...) {
        m.tags.clear();
    }
    m.model_type = col_text(7);
    m.file_size = static_cast<size_t>(sqlite3_column_int64(stmt, 8));
    m.checksum_sha256 = col_text(9);
    try {
        m.config = nlohmann::json::parse(col_text(10));
    } catch (...) {
        m.config = nlohmann::json::object();
    }
    m.created_at = col_text(11);
    m.updated_at = col_text(12);
    m.download_count = sqlite3_column_int(stmt, 13);
    m.rating = static_cast<float>(sqlite3_column_double(stmt, 14));
    m.is_public = sqlite3_column_int(stmt, 15) != 0;
    return m;
}

bool MetadataManager::bind_and_step_insert(sqlite3_stmt *stmt, const ModelMetadata &m)
{
    sqlite3_bind_text(stmt, 1, m.id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, m.name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, m.description.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, m.author.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, m.version.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 6, m.license.c_str(), -1, SQLITE_TRANSIENT);
    const std::string tags_json = nlohmann::json(m.tags).dump();
    sqlite3_bind_text(stmt, 7, tags_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 8, m.model_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 9, static_cast<sqlite3_int64>(m.file_size));
    sqlite3_bind_text(stmt, 10, m.checksum_sha256.c_str(), -1, SQLITE_TRANSIENT);
    const std::string config_json = m.config.dump();
    sqlite3_bind_text(stmt, 11, config_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 12, m.created_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 13, m.updated_at.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 14, m.download_count);
    sqlite3_bind_double(stmt, 15, m.rating);
    sqlite3_bind_int(stmt, 16, m.is_public ? 1 : 0);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        spdlog::error("MetadataManager: insert failed for '{}': {}", m.id, sqlite3_errmsg(db_));
        return false;
    }
    return true;
}

#endif  // SERIUS_HUB_SQLITE

}  // namespace serius::hub
