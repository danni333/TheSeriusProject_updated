#include "cli/cmd_mcp.h"

#include <httplib.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include "../mcp/audit.h"
#include "../mcp/server.h"

namespace serius {
namespace cli {

namespace {

void register_builtin_tools(mcp::MCPServer &server)
{
    server.register_tool(
        "echo", "Echo back the provided message",
        mcp::json{{"type", "object"},
                  {"properties", mcp::json{{"message", mcp::json{{"type", "string"}}}}},
                  {"required", mcp::json::array({"message"})}},
        [](const mcp::json &args) { return mcp::json{{"message", args.value("message", std::string{})}}; });

    server.register_tool("current_time", "Return the current UTC time", mcp::json::object(), [](const mcp::json &) {
        const std::time_t now = std::time(nullptr);
        std::tm tm{};
#if defined(_MSC_VER)
        gmtime_s(&tm, &now);
#else
            gmtime_r(&now, &tm);
#endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
        return mcp::json{{"utc", buf}, {"unix", static_cast<long long>(now)}};
    });

    server.register_tool(
        "serius_info", "Return Serius engine version and MCP server info", mcp::json::object(), [](const mcp::json &) {
            return mcp::json{{"engine", "TheSeriusProject"}, {"version", "0.1.0"}, {"protocol", "2024-11-05"}};
        });
}

}  // namespace

int cmd_mcp_serve(int argc, char **argv)
{
    std::string audit_path = "mcp_audit.log";
    int http_port = 0;  // 0 => stdio transport (default MCP mode)

    // CLI::run shifts argv: argv[0] is the command name, args start at 1.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--audit" && i + 1 < argc) {
            audit_path = argv[++i];
        } else if (arg == "--http" && i + 1 < argc) {
            http_port = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: serius mcp-serve [--audit PATH] [--http PORT]\n"
                         "  Default: serve MCP over stdio (newline-delimited JSON-RPC).\n"
                         "  --http PORT: also/instead expose POST /mcp over HTTP.\n";
            return 0;
        }
    }

    mcp::MCPServer server;
    register_builtin_tools(server);

    mcp::Audit audit(audit_path);
    server.set_audit(&audit);

    spdlog::info("MCP server starting ({} tool(s))", 3);

    if (http_port > 0) {
        httplib::Server http;
        server.register_http_route(http, "/mcp");
        http.Get("/health", [](const httplib::Request &, httplib::Response &res) {
            res.set_content(mcp::json{{"status", "ok"}}.dump(), "application/json");
        });
        spdlog::info("MCP HTTP endpoint on port {} (POST /mcp)", http_port);
        return http.listen("0.0.0.0", http_port);
    }

    return server.serve_stdio();
}

}  // namespace cli
}  // namespace serius
