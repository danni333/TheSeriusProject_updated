#pragma once

#include <chrono>
#include <ctime>
#include <string>
#include <vector>

#include "hub_types.h"

#ifdef SERIUS_HUB_SQLITE
struct sqlite3;
struct sqlite3_stmt;
#endif

namespace serius::hub {

class MetadataManager
{
   public:
    MetadataManager(const std::string &db_path);
    ~MetadataManager();

    bool initialize();
    bool save_model_metadata(const ModelMetadata &metadata);
    bool update_model_metadata(const std::string &model_id, const ModelMetadata &metadata);
    ModelMetadata get_model_metadata(const std::string &model_id);
    /// All stored models (used by SearchEngine semantic/similarity paths).
    std::vector<ModelMetadata> list_all_models();
    std::vector<ModelMetadata> search_models(const std::string &query = "", const std::vector<std::string> &tags = {},
                                             const std::string &author = "", int limit = 50, int offset = 0);
    bool delete_model_metadata(const std::string &model_id);

    // Statistics
    int get_model_count();
    int get_download_count(const std::string &model_id);
    bool increment_download_count(const std::string &model_id);
    bool update_rating(const std::string &model_id, float rating);

   private:
#ifdef SERIUS_HUB_SQLITE
    sqlite3 *db_ = nullptr;
#endif
    std::string db_path_;
    // Set when a legacy JSON file could not be migrated (e.g. rename failed):
    // the JSON helpers stay authoritative for this instance.
    bool use_json_fallback_ = false;

    bool create_tables();
    bool execute_query(const std::string &sql);
#ifdef SERIUS_HUB_SQLITE
    bool ensure_open();
    bool migrate_json_if_needed();
    ModelMetadata row_to_model(sqlite3_stmt *stmt);
    bool bind_and_step_insert(sqlite3_stmt *stmt, const ModelMetadata &m);
#endif
    // ModelMetadata row_to_model(sqlite3_stmt *stmt);
    std::vector<ModelMetadata> execute_search_query(const std::string &sql, const std::vector<std::string> &params);

    // File-based JSON storage helpers (db_path_ holds {id: model_json}).
    // Used as fallback when SQLite is unavailable and for one-time migration
    // of legacy JSON metadata files.
    nlohmann::json load_db() const;
    bool save_db(const nlohmann::json &db);
};

/// Current UTC time as ISO-8601 (used for created_at/updated_at).
inline std::string now_iso8601()
{
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
#if defined(_MSC_VER)
    struct tm tm_buf;
    gmtime_s(&tm_buf, &t);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
#else
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
#endif
    return buf;
}

}  // namespace serius::hub
