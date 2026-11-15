#include "server.h"

#include "audit.h"

#include <httplib.h>
#include <spdlog/spdlog.h>

#include <iostream>

namespace serius {
namespace mcp {

using json = nlohmann::json;

namespace {

json make_error(const json &id, int code, const std::string &message)
{
    return json{{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", code}, {"message", message}}}};
}

json make_result(const json &id, const json &result)
{
    return json{{"jsonrpc", "2.0"}, {"id", id}, {"result", result}};
}

}  // namespace

void MCPServer::register_tool(const std::string &name, const std::string &description,
                              const json &input_schema, ToolHandler handler)
{
    tools_[name] = ToolDef{description, input_schema.is_object() ? input_schema : json::object(),
                           std::move(handler)};
}

json MCPServer::handle_initialize() const
{
    return json{{"protocolVersion", "2024-11-05"},
                {"capabilities", json{{"tools", json::object()}}},
                {"serverInfo", json{{"name", "serius-mcp"}, {"version", "0.1.0"}}}};
}

json MCPServer::handle_tools_list() const
{
    json arr = json::array();
    for (const auto &[name, tool] : tools_) {
        arr.push_back(json{{"name", name},
                           {"description", tool.description},
                           {"inputSchema", tool.input_schema}});
    }
    return json{{"tools", arr}};
}

json MCPServer::handle_tools_call(const json &params, const json &id)
{
    const std::string name = params.value("name", std::string{});
    if (name.empty()) {
        return make_error(id, -32602, "tools/call requires 'name'");
    }
    auto it = tools_.find(name);
    if (it == tools_.end()) {
        return make_error(id, -32602, "unknown tool: " + name);
    }

    const json args = params.contains("arguments") && params["arguments"].is_object()
                          ? params["arguments"]
                          : json::object();

    std::string outcome = "ok";
    json result;
    try {
        result = it->second.handler(args);
        if (result.is_discarded()) { result = json::object(); }
    } catch (const std::exception &e) {
        outcome = std::string("error: ") + e.what();
        result = json{{"error", e.what()}};
    }

    if (audit_) {
        audit_->log("tools/call:" + name, args.dump(), outcome);
    }

    return make_result(id, json{{"content",
                                 json::array({json{{"type", "text"},
                                                   {"text", result.dump()}}})},
                                {"structuredContent", result},
                                {"isError", outcome != "ok"}});
}

json MCPServer::handle_message(const json &request)
{
    if (!request.is_object() || request.value("jsonrpc", "") != "2.0" ||
        !request.contains("method") || !request["method"].is_string()) {
        return make_error(json(nullptr), -32600, "Invalid JSON-RPC request");
    }

    const bool is_notification = !request.contains("id");
    const json id = is_notification ? json(nullptr) : request["id"];
    const std::string method = request["method"].get<std::string>();
    const json params =
        request.contains("params") && request["params"].is_object() ? request["params"]
                                                                      : json::object();

    if (is_notification) {
        // Notifications (e.g. notifications/initialized) get no response.
        return json(json::value_t::discarded);
    }

    try {
        if (method == "initialize") {
            return make_result(id, handle_initialize());
        }
        if (method == "ping") {
            return make_result(id, json::object());
        }
        if (method == "tools/list") {
            return make_result(id, handle_tools_list());
        }
        if (method == "tools/call") {
            return handle_tools_call(params, id);
        }
        return make_error(id, -32601, "Method not found: " + method);
    } catch (const std::exception &e) {
        spdlog::warn("MCP server error handling '{}': {}", method, e.what());
        return make_error(id, -32603, e.what());
    }
}

void MCPServer::register_http_route(httplib::Server &server,
                                    const std::string &path)
{
    server.Post(path, [this](const httplib::Request &req, httplib::Response &res) {
        json request;
        try {
            request = json::parse(req.body);
        } catch (const std::exception &e) {
            res.status = 400;
            res.set_content(make_error(json(nullptr), -32700,
                                       std::string("Parse error: ") + e.what())
                                .dump(),
                            "application/json");
            return;
        }
        json response = handle_message(request);
        if (response.is_discarded()) {
            res.status = 202;  // notification: accepted, no body
            return;
        }
        res.set_content(response.dump(), "application/json");
    });
}

int MCPServer::serve_stdio()
{
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) { continue; }
        json request;
        try {
            request = json::parse(line);
        } catch (const std::exception &e) {
            json err = make_error(json(nullptr), -32700,
                                  std::string("Parse error: ") + e.what());
            std::cout << err.dump() << "\n" << std::flush;
            continue;
        }
        json response = handle_message(request);
        if (!response.is_discarded()) {
            std::cout << response.dump() << "\n" << std::flush;
        }
    }
    return 0;
}

}  // namespace mcp
}  // namespace serius
