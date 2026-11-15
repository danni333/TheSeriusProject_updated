#pragma once

#include "hub_types.h"
// #include <sqlite3.h>  // Requires SQLite - disabled for now

#include <chrono>
#include <ctime>

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
    std::vector<ModelMetadata> search_models(const std::string &query = "", const std::vector<std::string> &tags = {},
                                             const std::string &author = "", int limit = 50, int offset = 0);
    bool delete_model_metadata(const std::string &model_id);

    // Statistics
    int get_model_count();
    int get_download_count(const std::string &model_id);
    bool increment_download_count(const std::string &model_id);
    bool update_rating(const std::string &model_id, float rating);

   private:
    // sqlite3 *db_;
    std::string db_path_;

    bool create_tables();
    bool execute_query(const std::string &sql);
    // ModelMetadata row_to_model(sqlite3_stmt *stmt);
    std::vector<ModelMetadata> execute_search_query(const std::string &sql, const std::vector<std::string> &params);

    // File-based JSON storage helpers (db_path_ holds {id: model_json}).
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
