#include "model_hub_server.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>


namespace serius::hub {

ModelHubServer::ModelHubServer(int port) : port_(port), models_dir_("models"), metadata_db_path_("models_metadata.json")
{
    server_ = std::make_unique<httplib::Server>();
    std::filesystem::create_directories(models_dir_);
}

ModelHubServer::~ModelHubServer()
{
    stop();
}

void ModelHubServer::start()
{
    setup_routes();
    spdlog::info("Model Hub Server starting on port {}", port_);
    server_->listen("0.0.0.0", port_);
}

void ModelHubServer::stop()
{
    if (server_) {
        server_->stop();
    }
}

void ModelHubServer::setup_routes()
{
    setup_model_routes();
    setup_auth_routes();
    setup_search_routes();
}

void ModelHubServer::setup_model_routes()
{
    // Upload model
    server_->Post("/v1/models/upload", [this](const httplib::Request &req, httplib::Response &res) {
        std::string user_id;
        if (!authenticate_user(req.get_header_value("Authorization"), user_id)) {
            res.status = 401;
            return;
        }

        if (!req.has_file("model") || !req.has_file("metadata")) {
            res.status = 400;
            return;
        }

        const auto &model_file = req.get_file_value("model");
        const auto &metadata_file = req.get_file_value("metadata");

        try {
            auto metadata = json_to_model(nlohmann::json::parse(metadata_file.content));
            metadata.id = generate_model_id(metadata.name, metadata.author);
            metadata.file_size = model_file.content.size();
            metadata.checksum_sha256 = calculate_checksum(model_file.filename);

            std::string model_path = models_dir_ + "/" + metadata.id + "_" + model_file.filename;
            std::ofstream out(model_path, std::ios::binary);
            out.write(model_file.content.data(), model_file.content.size());

            res.status = 201;
            res.set_content(model_to_json(metadata).dump(), "application/json");
        }
        catch (const std::exception &e) {
            res.status = 500;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // Download model
    server_->Get("/v1/models/(.*)/download", [this](const httplib::Request &req, httplib::Response &res) {
        std::string model_id = req.matches[1];

        auto metadata = get_model_metadata(model_id);
        if (metadata.id.empty()) {
            res.status = 404;
            return;
        }

        std::string model_path = models_dir_ + "/" + model_id + "_model.gguf";
        if (!std::filesystem::exists(model_path)) {
            res.status = 404;
            return;
        }

        res.set_content_provider(metadata.file_size, "application/octet-stream",
                                 [model_path](size_t offset, size_t length, httplib::DataSink &sink) {
                                     std::ifstream file(model_path, std::ios::binary);
                                     file.seekg(offset);
                                     std::vector<char> buffer(length);
                                     file.read(buffer.data(), length);
                                     sink.write(buffer.data(), file.gcount());
                                     return true;
                                 });
    });
}

void ModelHubServer::setup_auth_routes() {}
void ModelHubServer::setup_search_routes() {}

std::vector<ModelMetadata> ModelHubServer::search_models(const std::string &query, const std::vector<std::string> &tags,
                                                         int limit, int offset)
{
    return {};
}

ModelMetadata ModelHubServer::get_model_metadata(const std::string &model_id)
{
    return ModelMetadata{};
}

bool ModelHubServer::authenticate_user(const std::string &token, std::string &user_id)
{
    user_id = "default_user";
    return true;
}

std::string ModelHubServer::generate_model_id(const std::string &name, const std::string &author)
{
    return name + "-" + author;
}

nlohmann::json ModelHubServer::model_to_json(const ModelMetadata &metadata)
{
    return serius::hub::model_to_json(metadata);
}

ModelMetadata ModelHubServer::json_to_model(const nlohmann::json &json)
{
    return serius::hub::json_to_model(json);
}

std::string ModelHubServer::calculate_checksum(const std::string &file_path)
{
    // Simple checksum implementation for now
    std::ifstream file(file_path, std::ios::binary);
    if (!file) return "";

    std::stringstream ss;
    ss << file.rdbuf();
    std::string content = ss.str();

    size_t hash = 0;
    for (char c : content) {
        hash = hash * 31 + c;
    }

    std::stringstream result;
    result << std::hex << hash;
    return result.str();
}

}  // namespace serius::hub
