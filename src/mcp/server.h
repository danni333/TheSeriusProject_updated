#pragma once

#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <string>

namespace httplib {
class Server;
}

namespace serius {
namespace mcp {

using json = nlohmann::json;

class Audit;

/// Handler invoked for a `tools/call` request. Receives the tool's
/// `arguments` object and returns a JSON result (wrapped in a
/// spec-shaped `content`/`structuredContent` envelope by the server).
using ToolHandler = std::function<json(const json &arguments)>;

/// Minimal MCP (Model Context Protocol) server: JSON-RPC 2.0 dispatch for
/// `initialize`, `ping`, `tools/list` and `tools/call`.
///
/// Transports:
///  - `serve_stdio()`: newline-delimited JSON-RPC on stdin/stdout (MCP stdio).
///  - `register_http_route(httplib::Server&, path)`: POST endpoint for the
///    HTTP transport (`HttpTransport`).
///  - `handle_message()`: pure dispatch, usable in-process or in tests.
class MCPServer
{
   public:
    MCPServer() = default;

    /// Register a tool. `input_schema` is a JSON Schema object describing
    /// the arguments (may be empty).
    void register_tool(const std::string &name, const std::string &description, const json &input_schema,
                       ToolHandler handler);

    /// Attach an audit log; every `tools/call` is recorded.
    void set_audit(Audit *audit) { audit_ = audit; }

    /// Dispatch one JSON-RPC message. Returns the response object, or an
    /// empty json for notifications (messages without "id").
    json handle_message(const json &request);

    /// Register a POST route on an existing httplib::Server.
    void register_http_route(httplib::Server &server, const std::string &path = "/mcp");

    /// Blocking newline-delimited JSON-RPC loop on stdin/stdout.
    /// Returns when stdin closes.
    int serve_stdio();

   private:
    struct ToolDef
    {
        std::string description;
        json input_schema;
        ToolHandler handler;
    };

    std::map<std::string, ToolDef> tools_;
    Audit *audit_ = nullptr;

    json handle_initialize() const;
    json handle_tools_list() const;
    json handle_tools_call(const json &params, const json &id);
};

}  // namespace mcp
}  // namespace serius
