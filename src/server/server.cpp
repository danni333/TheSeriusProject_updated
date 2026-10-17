#include "server.h"

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace serius {
namespace server {

namespace {

static std::chrono::steady_clock::time_point g_start_time;
static std::atomic<uint64_t> g_request_count = 0;

std::string extract_user_prompt(const nlohmann::json& body)
{
    const auto messages = body.value("messages", nlohmann::json::array());
    if (!messages.is_array()) {
        return "Serius API is running. Load a model to get real completions.";
    }

    std::string last_user_text;
    for (const auto& msg : messages) {
        if (!msg.is_object()) {
            continue;
        }

        const std::string role = msg.value("role", "");
        if (role == "user") {
            const auto content = msg.value("content", nlohmann::json());
            if (content.is_string()) {
                last_user_text = content.get<std::string>();
            }
        }
    }

    if (!last_user_text.empty()) {
        return last_user_text;
    }

    for (const auto& msg : messages) {
        if (!msg.is_object()) {
            continue;
        }

        const auto content = msg.value("content", nlohmann::json());
        if (content.is_string()) {
            const std::string candidate = content.get<std::string>();
            if (!candidate.empty()) {
                return candidate;
            }
        }
    }

    return "Serius API is running. Load a model to get real completions.";
}

nlohmann::json build_chat_completion_response(const nlohmann::json& body, const std::string& content)
{
    const std::string model = body.value("model", "default");
    const auto created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());

    nlohmann::json out;
    out["id"] = "serius-stub";
    out["object"] = "chat.completion";
    out["created"] = created;
    out["model"] = model;
    out["choices"] = nlohmann::json::array();
    out["choices"][0] = {
        {"index", 0},
        {"message", {{"role", "assistant"}, {"content", content}}},
        {"finish_reason", "stop"}};
    out["usage"] = {{"prompt_tokens", 0}, {"completion_tokens", 1}, {"total_tokens", 1}};
    return out;
}

std::string build_sse_stream(const nlohmann::json& body, const std::string& content)
{
    const std::string model = body.value("model", "default");
    const auto created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());

    nlohmann::json intro = {
        {"id", "serius-stub"},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", {{
            {"index", 0},
            {"delta", {{"role", "assistant"}}},
            {"finish_reason", nullptr}
        }}}
    };

    nlohmann::json content_chunk = {
        {"id", "serius-stub"},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", {{
            {"index", 0},
            {"delta", {{"content", content}}},
            {"finish_reason", nullptr}
        }}}
    };

    nlohmann::json end_chunk = {
        {"id", "serius-stub"},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", {{
            {"index", 0},
            {"delta", {}},
            {"finish_reason", "stop"}
        }}}
    };

    std::string stream;
    stream += "data: " + intro.dump() + "\n\n";
    stream += "data: " + content_chunk.dump() + "\n\n";
    stream += "data: " + end_chunk.dump() + "\n\n";
    stream += "data: [DONE]\n\n";
    return stream;
}

void setup_routes(httplib::Server& srv)
{
    g_start_time = std::chrono::steady_clock::now();

    srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    srv.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        const auto now = std::chrono::steady_clock::now();
        const auto uptime_sec = std::chrono::duration<double>(now - g_start_time).count();
        nlohmann::json j;
        j["uptime_seconds"] = uptime_sec;
        j["request_count"] = g_request_count.load(std::memory_order_relaxed);
        res.set_content(j.dump(), "application/json");
    });

    srv.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        g_request_count.fetch_add(1, std::memory_order_relaxed);
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const std::exception&) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"Invalid JSON\"}}", "application/json");
            return;
        }

        const std::string prompt = extract_user_prompt(body);
        const std::string completion = "Serius API is running. Load a model to get real completions.";
        const std::string assistant_text = prompt.empty() ? completion : prompt;
        const bool stream = body.value("stream", false);

        if (stream) {
            res.status = 200;
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_content(build_sse_stream(body, assistant_text), "text/event-stream");
            return;
        }

        res.set_content(build_chat_completion_response(body, assistant_text).dump(), "application/json");
    });

    srv.Post("/v1/completions", [](const httplib::Request& req, httplib::Response& res) {
        g_request_count.fetch_add(1, std::memory_order_relaxed);
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const std::exception&) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"Invalid JSON\"}}", "application/json");
            return;
        }

        const std::string prompt = body.value("prompt", "");
        const std::string content = prompt.empty() ? "Serius API is running. Load a model to get real completions." : prompt;
        nlohmann::json out;
        out["id"] = "serius-stub";
        out["object"] = "text_completion";
        out["model"] = body.value("model", "default");
        out["choices"] = nlohmann::json::array();
        out["choices"][0] = {{"text", content}, {"index", 0}, {"finish_reason", "stop"}};
        out["usage"] = {{"prompt_tokens", 0}, {"completion_tokens", 1}, {"total_tokens", 1}};
        res.set_content(out.dump(), "application/json");
    });

    srv.Post("/v1/embeddings", [](const httplib::Request& req, httplib::Response& res) {
        g_request_count.fetch_add(1, std::memory_order_relaxed);
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const std::exception&) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"Invalid JSON\"}}", "application/json");
            return;
        }

        nlohmann::json out;
        out["object"] = "list";
        out["data"] = nlohmann::json::array();
        out["data"][0] = {{"object", "embedding"}, {"embedding", nlohmann::json::array({0.0f, 0.0f, 0.0f})}, {"index", 0}};
        out["model"] = body.value("model", "default");
        out["usage"] = {{"prompt_tokens", 0}, {"total_tokens", 0}};
        res.set_content(out.dump(), "application/json");
    });

    srv.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        g_request_count.fetch_add(1, std::memory_order_relaxed);
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
    setup_routes(server_);
    spdlog::info("Serius server listening on 0.0.0.0:{}", port_);
    const bool ok = server_.listen("0.0.0.0", static_cast<int>(port_));
    if (!ok) {
        spdlog::error("Failed to listen on port {}", port_);
        return 1;
    }
    return 0;
}

void Server::stop()
{
    server_.stop();
}

}  // namespace server
}  // namespace serius
