#pragma once

#include <httplib.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "hub_types.h"
#include "metadata_manager.h"
#include "search_engine.h"
#include "../common/defaults.h"

namespace serius::hub {

class ModelHubServer
{
   public:
    ModelHubServer(int port = defaults::kHubPort);
    ~ModelHubServer();

    void start();
    void stop();
    void setup_routes();

    // Model management
    bool upload_model(const std::string &model_path, const ModelMetadata &metadata);
    bool download_model(const std::string &model_id, const std::string &output_path);
    std::vector<ModelMetadata> search_models(const std::string &query,
                                             const std::vector<std::string> &tags = {},
                                             int limit = 50, int offset = 0);
    bool delete_model(const std::string &model_id);
    ModelMetadata get_model_metadata(const std::string &model_id);

    // Authentication (file-based API keys in hub_users.json)
    bool authenticate_user(const std::string &token, std::string &user_id);
    bool check_permissions(const std::string &user_id, const std::string &action,
                           const std::string &resource = "");

   private:
    std::unique_ptr<httplib::Server> server_;
    int port_;
    std::string models_dir_;
    std::string metadata_db_path_;
    std::string users_path_;

    MetadataManager metadata_;
    SearchEngine search_;
    std::mutex db_mutex_;

    // Route handlers
    void setup_model_routes();
    void setup_auth_routes();
    void setup_search_routes();

    // Helper methods
    std::string generate_model_id(const std::string &name, const std::string &author);
    std::string calculate_checksum(const std::string &file_path);
    std::string calculate_checksum_bytes(const std::string &content);
    nlohmann::json model_to_json(const ModelMetadata &metadata);
    ModelMetadata json_to_model(const nlohmann::json &json);
    bool validate_model_file(const std::string &file_path);
    void log_access(const std::string &user_id, const std::string &action,
                    const std::string &resource);

    // File-based user store: {api_key: {user_id, username, permissions[]}}
    nlohmann::json load_users() const;
    bool save_users(const nlohmann::json &users);
    std::string register_user(const std::string &username);
    std::vector<std::string> user_permissions(const std::string &user_id) const;
    std::string resolve_user_id(const std::string &api_key) const;
};

}  // namespace serius::hub
