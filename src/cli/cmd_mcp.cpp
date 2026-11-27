#include "cli/cmd_mcp.h"

#include <httplib.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include "../mcp/audit.h"
#include "../mcp/server.h"

#ifndef SERIUS_VERSION
#define SERIUS_VERSION "unknown"
#endif

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
            return mcp::json{{"engine", "TheSeriusProject"}, {"version", SERIUS_VERSION}, {"protocol", "2024-11-05"}};
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
        // Optional bearer auth: when SERIUS_MCP_TOKEN is set, /mcp requires
        // "Authorization: Bearer <token>". Unset keeps the open local mode.
        const char *mcp_token = std::getenv("SERIUS_MCP_TOKEN");
        if (mcp_token && *mcp_token) {
            const std::string expected = mcp_token;
            http.set_pre_routing_handler([expected](const httplib::Request &req, httplib::Response &res) {
                if (req.path == "/health") {
                    return httplib::Server::HandlerResponse::Unhandled;
                }
                const std::string auth = req.get_header_value("Authorization");
                const std::string prefix = "Bearer ";
                const bool ok = auth.size() == expected.size() + prefix.size() &&
                                auth.compare(0, prefix.size(), prefix) == 0;
                uint8_t diff = ok ? 0 : 1;
                if (ok) {
                    for (size_t i = 0; i < expected.size(); ++i) {
                        diff |= static_cast<uint8_t>(auth[prefix.size() + i] ^ expected[i]);
                    }
                }
                if (diff != 0) {
                    res.status = 401;
                    res.set_content("{\"error\":\"unauthorized\"}", "application/json");
                    return httplib::Server::HandlerResponse::Handled;
                }
                return httplib::Server::HandlerResponse::Unhandled;
            });
        }
        spdlog::info("MCP HTTP endpoint on port {} (POST /mcp)", http_port);
        return http.listen("0.0.0.0", http_port);
    }

    return server.serve_stdio();
}

}  // namespace cli
}  // namespace serius
