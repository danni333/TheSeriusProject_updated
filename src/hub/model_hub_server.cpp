#include "model_hub_server.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>
#include <spdlog/spdlog.h>

namespace serius::hub {

ModelHubServer::ModelHubServer(int port) 
    : port_(port), models_dir_("models"), metadata_db_path_("models_metadata.json") {
    server_ = std::make_unique<httplib::Server>();
    std::filesystem::create_directories(models_dir_);
}

void ModelHubServer::setup_routes() {
    setup_model_routes();
    setup_auth_routes();
    setup_search_routes();
}

void ModelHubServer::setup_model_routes() {
    // Upload model
    server_->Post("/v1/models/upload", [this](const httplib::Request& req, httplib::Response& res) {
        std::string user_id;
        if (!authenticate_user(req.get_header_value("Authorization"), user_id)) {
            res.status = 401;
            return;
        }

        if (!req.has_file("model") || !req.has_file("metadata")) {
            res.status = 400;
            return;
        }

        const auto& model_file = req.get_file_value("model");
        const auto& metadata_file = req.get_file_value("metadata");

        try {
            auto metadata = json_to_model(nlohmann::json::parse(metadata_file.content));
            metadata.id = generate_model_id(metadata.name, metadata.author);
            metadata.file_size = model_file.length;
            metadata.checksum_sha256 = calculate_checksum(model_file.filename);

            std::string model_path = models_dir_ + "/" + metadata.id + "_" + model_file.filename;
            std::ofstream out(model_path, std::ios::binary);
            out.write(model_file.content.data(), model_file.content.size());

            res.status = 201;
            res.set_content(model_to_json(metadata).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // Download model
    server_->Get("/v1/models/(.*)/download", [this](const httplib::Request& req, httplib::Response& res) {
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

        res.set_content_provider(
            model_path,
            "application/octet-stream",
            [model_path](size_t offset, size_t length, httplib::DataSink& sink) {
                std::ifstream file(model_path, std::ios::binary);
                file.seekg(offset);
                std::vector<char> buffer(length);
                file.read(buffer.data(), length);
                sink.write(buffer.data(), file.gcount());
                return true;
            }
        );
    });
}

std::string ModelHubServer::calculate_checksum(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    std::vector<char> buffer(4096);
    
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    
    while (file.read(buffer.data(), buffer.size())) {
        SHA256_Update(&ctx, buffer.data(), file.gcount());
    }
    
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &ctx);
    
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    
    return ss.str();
}

} // namespace serius::hub
