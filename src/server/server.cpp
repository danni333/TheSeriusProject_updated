#include "server.h"

#include <chrono>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace serius {
namespace server {

namespace {

static std::chrono::steady_clock::time_point g_start_time;
static uint64_t g_request_count = 0;

void setup_routes(httplib::Server &srv)
{
    g_start_time = std::chrono::steady_clock::now();

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
