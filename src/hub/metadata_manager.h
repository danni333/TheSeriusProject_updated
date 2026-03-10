#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <sqlite3.h>

namespace serius::hub {

class MetadataManager {
public:
    MetadataManager(const std::string& db_path);
    ~MetadataManager();

    bool initialize();
    bool save_model_metadata(const ModelMetadata& metadata);
    bool update_model_metadata(const std::string& model_id, const ModelMetadata& metadata);
    ModelMetadata get_model_metadata(const std::string& model_id);
    std::vector<ModelMetadata> search_models(const std::string& query = "",
                                           const std::vector<std::string>& tags = {},
                                           const std::string& author = "",
                                           int limit = 50, int offset = 0);
    bool delete_model_metadata(const std::string& model_id);
    
    // Statistics
    int get_model_count();
    int get_download_count(const std::string& model_id);
    bool increment_download_count(const std::string& model_id);
    bool update_rating(const std::string& model_id, float rating);

private:
    sqlite3* db_;
    std::string db_path_;

    bool create_tables();
    bool execute_query(const std::string& sql);
    ModelMetadata row_to_model(sqlite3_stmt* stmt);
    std::vector<ModelMetadata> execute_search_query(const std::string& sql, 
                                                   const std::vector<std::string>& params);
};

} // namespace serius::hub
