#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace serius::hub {

struct ModelMetadata {
    std::string id;
    std::string name;
    std::string description;
    std::string author;
    std::string version;
    std::string license;
    std::vector<std::string> tags;
    std::string model_type;  // "gguf", "safetensors", etc.
    size_t file_size;
    std::string checksum_sha256;
    nlohmann::json config;
    std::string created_at;
    std::string updated_at;
    int download_count;
    float rating;
    bool is_public;
};

class ModelHubServer {
public:
    ModelHubServer(int port = 8080);
    ~ModelHubServer();

    void start();
    void stop();
    void setup_routes();

    // Model management
    bool upload_model(const std::string& model_path, const ModelMetadata& metadata);
    bool download_model(const std::string& model_id, const std::string& output_path);
    std::vector<ModelMetadata> search_models(const std::string& query, 
                                            const std::vector<std::string>& tags = {},
                                            int limit = 50, int offset = 0);
    bool delete_model(const std::string& model_id);
    ModelMetadata get_model_metadata(const std::string& model_id);

    // Authentication
    bool authenticate_user(const std::string& token, std::string& user_id);
    bool check_permissions(const std::string& user_id, const std::string& action, const std::string& resource);

private:
    std::unique_ptr<httplib::Server> server_;
    int port_;
    std::string models_dir_;
    std::string metadata_db_path_;

    // Route handlers
    void setup_model_routes();
    void setup_auth_routes();
    void setup_search_routes();

    // Helper methods
    std::string generate_model_id(const std::string& name, const std::string& author);
    std::string calculate_checksum(const std::string& file_path);
    nlohmann::json model_to_json(const ModelMetadata& metadata);
    ModelMetadata json_to_model(const nlohmann::json& json);
    bool validate_model_file(const std::string& file_path);
    void log_access(const std::string& user_id, const std::string& action, const std::string& resource);
};

} // namespace serius::hub
