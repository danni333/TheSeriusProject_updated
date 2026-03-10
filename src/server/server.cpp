#include "server.h"
#include "../hub/model_hub_server.h"

#include <chrono>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace serius {
namespace server {

namespace {

static std::chrono::steady_clock::time_point g_start_time;
static uint64_t g_request_count = 0;
static std::unique_ptr<serius::hub::ModelHubServer> g_hub_server;

void setup_hub_routes(httplib::Server &srv)
{
    if (!g_hub_server) {
        g_hub_server = std::make_unique<serius::hub::ModelHubServer>(8081);
    }

    // Model upload endpoint
    srv.Post("/v1/hub/models/upload", [](const httplib::Request &req, httplib::Response &res) {
        if (!g_hub_server) {
            res.status = 503;
            res.set_content("{\"error\":{\"message\":\"Hub service unavailable\"}}", "application/json");
            return;
        }
        
        // Forward to hub server
        res.status = 501;  // Not implemented yet
        res.set_content("{\"error\":{\"message\":\"Upload endpoint coming soon\"}}", "application/json");
    });

    // Model download endpoint
    srv.Get("/v1/hub/models/(.*)/download", [](const httplib::Request &req, httplib::Response &res) {
        if (!g_hub_server) {
            res.status = 503;
            res.set_content("{\"error\":{\"message\":\"Hub service unavailable\"}}", "application/json");
            return;
        }
        
        std::string model_id = req.matches[1];
        res.status = 501;  // Not implemented yet
        res.set_content("{\"error\":{\"message\":\"Download endpoint coming soon\"}}", "application/json");
    });

    // Search models endpoint
    srv.Get("/v1/hub/models/search", [](const httplib::Request &req, httplib::Response &res) {
        if (!g_hub_server) {
            res.status = 503;
            res.set_content("{\"error\":{\"message\":\"Hub service unavailable\"}}", "application/json");
            return;
        }
        
        std::string query = req.get_param_value("q");
        std::string tags_str = req.get_param_value("tags");
        
        // Parse tags
        std::vector<std::string> tags;
        if (!tags_str.empty()) {
            std::stringstream ss(tags_str);
            std::string tag;
            while (std::getline(ss, tag, ',')) {
                tags.push_back(tag);
            }
        }
        
        auto results = g_hub_server->search_models(query, tags, 50, 0);
        
        nlohmann::json response;
        response["object"] = "list";
        response["data"] = nlohmann::json::array();
        
        for (const auto& model : results) {
            nlohmann::json model_json;
            model_json["id"] = model.id;
            model_json["name"] = model.name;
            model_json["description"] = model.description;
            model_json["author"] = model.author;
            model_json["tags"] = model.tags;
            model_json["download_count"] = model.download_count;
            model_json["rating"] = model.rating;
            response["data"].push_back(model_json);
        }
        
        res.set_content(response.dump(), "application/json");
    });

    // Get model metadata
    srv.Get("/v1/hub/models/(.*)", [](const httplib::Request &req, httplib::Response &res) {
        if (!g_hub_server) {
            res.status = 503;
            res.set_content("{\"error\":{\"message\":\"Hub service unavailable\"}}", "application/json");
            return;
        }
        
        std::string model_id = req.matches[1];
        auto metadata = g_hub_server->get_model_metadata(model_id);
        
        if (metadata.id.empty()) {
            res.status = 404;
            res.set_content("{\"error\":{\"message\":\"Model not found\"}}", "application/json");
            return;
        }
        
        nlohmann::json response;
        response["id"] = metadata.id;
        response["name"] = metadata.name;
        response["description"] = metadata.description;
        response["author"] = metadata.author;
        response["version"] = metadata.version;
        response["license"] = metadata.license;
        response["tags"] = metadata.tags;
        response["model_type"] = metadata.model_type;
        response["file_size"] = metadata.file_size;
        response["checksum_sha256"] = metadata.checksum_sha256;
        response["config"] = metadata.config;
        response["created_at"] = metadata.created_at;
        response["updated_at"] = metadata.updated_at;
        response["download_count"] = metadata.download_count;
        response["rating"] = metadata.rating;
        response["is_public"] = metadata.is_public;
        
        res.set_content(response.dump(), "application/json");
    });
}

void setup_routes(httplib::Server &srv)
{
    g_start_time = std::chrono::steady_clock::now();
    
    // Setup hub routes first
    setup_hub_routes(srv);

    srv.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    srv.Get("/metrics", [](const httplib::Request &, httplib::Response &res) {
        auto now = std::chrono::steady_clock::now();
        auto uptime_sec = std::chrono::duration<double>(now - g_start_time).count();
        nlohmann::json j;
        j["uptime_seconds"] = uptime_sec;
        j["request_count"] = g_request_count;
        res.set_content(j.dump(), "application/json");
    });

    srv.Post("/v1/chat/completions", [](const httplib::Request &req, httplib::Response &res) {
        g_request_count++;
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const std::exception &) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"Invalid JSON\"}}", "application/json");
            return;
        }

        std::string model = body.value("model", "default");

        nlohmann::json out;
        out["id"] = "serius-stub";
        out["object"] = "chat.completion";
        out["model"] = model;
        out["choices"] = nlohmann::json::array();
        out["choices"][0] = {
            {"index", 0},
            {"message", {{"role", "assistant"}, {"content", "Serius API is running. Load a model to get real completions."}}},
            {"finish_reason", "stop"}};
        out["usage"] = {{"prompt_tokens", 0}, {"completion_tokens", 1}, {"total_tokens", 1}};
        res.set_content(out.dump(), "application/json");
    });

    srv.Get("/v1/models", [](const httplib::Request &, httplib::Response &res) {
        g_request_count++;
        nlohmann::json out;
        out["object"] = "list";
        out["data"] = nlohmann::json::array();
        out["data"][0] = {{"id", "default"}, {"object", "model"}};
        res.set_content(out.dump(), "application/json");
    });
}

}  // namespace

Server::Server(uint16_t port) : port_(port) {}

int Server::run()
{
    httplib::Server srv;
    setup_routes(srv);
    spdlog::info("Serius server listening on 0.0.0.0:{}", port_);
    running_ = true;
    bool ok = srv.listen("0.0.0.0", static_cast<int>(port_));
    running_ = false;
    if (!ok) {
        spdlog::error("Failed to listen on port {}", port_);
        return 1;
    }
    return 0;
}

void Server::stop()
{
    // TODO: signal server thread to stop (requires background thread)
}

}  // namespace server
}  // namespace serius
