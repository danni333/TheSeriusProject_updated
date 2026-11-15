#include <gtest/gtest.h>

#include <filesystem>
#include <httplib.h>
#include <string>
#include <thread>

#include "mcp/client.h"
#include "mcp/server.h"
#include "mcp/transport_http.h"
#include "mcp/transport_stdio.h"

using namespace serius::mcp;

namespace {

MCPServer make_test_server()
{
    MCPServer server;
    server.register_tool(
        "echo",
        "Echo back the message",
        json{{"type", "object"},
             {"properties", json{{"message", json{{"type", "string"}}}}}},
        [](const json &args) { return json{{"message", args.value("message", "")} }; });
    server.register_tool("boom", "Always throws", json::object(),
                         [](const json &) -> json { throw std::runtime_error("kaput"); });
    return server;
}

json rpc(const std::string &method, int id, const json &params = json::object())
{
    return json{{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}};
}

// Locate the built `serius` CLI for the stdio end-to-end test.
std::filesystem::path find_serius_exe()
{
    if (const char *env = std::getenv("SERIUS_EXE")) {
        if (std::filesystem::exists(env)) { return env; }
    }
    const std::filesystem::path cwd = std::filesystem::current_path();
    const std::vector<std::filesystem::path> candidates = {
        cwd / "src" / "serius.exe",
        cwd / "serius.exe",
        cwd / ".." / "src" / "serius.exe",
        cwd / ".." / "serius.exe",
    };
    for (const auto &c : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(c, ec)) { return c; }
    }
    return {};
}

}  // namespace

// ---------- Unit: dispatch ----------

TEST(MCPServer, Initialize)
{
    MCPServer server = make_test_server();
    json resp = server.handle_message(rpc("initialize", 1));
    ASSERT_TRUE(resp.contains("result"));
    EXPECT_EQ(resp["result"]["serverInfo"]["name"], "serius-mcp");
    EXPECT_TRUE(resp["result"]["capabilities"].contains("tools"));
}

TEST(MCPServer, Ping)
{
    MCPServer server = make_test_server();
    json resp = server.handle_message(rpc("ping", 7));
    EXPECT_TRUE(resp.contains("result"));
    EXPECT_EQ(resp["id"], 7);
}

TEST(MCPServer, ToolsList)
{
    MCPServer server = make_test_server();
    json resp = server.handle_message(rpc("tools/list", 2));
    const auto &tools = resp["result"]["tools"];
    ASSERT_EQ(tools.size(), 2);
    EXPECT_EQ(tools[0]["name"], "boom");  // std::map ordering
    EXPECT_EQ(tools[1]["name"], "echo");
    EXPECT_TRUE(tools[1].contains("inputSchema"));
}

TEST(MCPServer, ToolsCallEcho)
{
    MCPServer server = make_test_server();
    json resp = server.handle_message(
        rpc("tools/call", 3, json{{"name", "echo"}, {"arguments", json{{"message", "hi"}}}}));
    ASSERT_TRUE(resp.contains("result"));
    EXPECT_FALSE(resp["result"].value("isError", true));
    EXPECT_EQ(resp["result"]["structuredContent"]["message"], "hi");
}

TEST(MCPServer, ToolsCallUnknownTool)
{
    MCPServer server = make_test_server();
    json resp = server.handle_message(
        rpc("tools/call", 4, json{{"name", "nope"}, {"arguments", json::object()}}));
    ASSERT_TRUE(resp.contains("error"));
    EXPECT_EQ(resp["error"]["code"], -32602);
}

TEST(MCPServer, ToolsCallHandlerError)
{
    MCPServer server = make_test_server();
    json resp = server.handle_message(
        rpc("tools/call", 5, json{{"name", "boom"}, {"arguments", json::object()}}));
    ASSERT_TRUE(resp.contains("result"));
    EXPECT_TRUE(resp["result"].value("isError", false));
    EXPECT_TRUE(resp["result"]["structuredContent"].contains("error"));
}

TEST(MCPServer, UnknownMethod)
{
    MCPServer server = make_test_server();
    json resp = server.handle_message(rpc("resources/list", 6));
    ASSERT_TRUE(resp.contains("error"));
    EXPECT_EQ(resp["error"]["code"], -32601);
}

TEST(MCPServer, InvalidRequest)
{
    MCPServer server = make_test_server();
    json resp = server.handle_message(json{{"foo", "bar"}});
    ASSERT_TRUE(resp.contains("error"));
    EXPECT_EQ(resp["error"]["code"], -32600);
}

TEST(MCPServer, NotificationGetsNoResponse)
{
    MCPServer server = make_test_server();
    json notif = json{{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}};
    EXPECT_TRUE(server.handle_message(notif).is_discarded());
}

// ---------- E2E: HTTP transport <-> MCPServer ----------

TEST(MCPServer, HttpEndToEnd)
{
    MCPServer server = make_test_server();
    httplib::Server http;
    server.register_http_route(http, "/mcp");

    const int port = http.bind_to_any_port("127.0.0.1");
    ASSERT_GE(port, 0);
    std::thread t([&http] { http.listen_after_bind(); });

    HttpTransport transport("http://127.0.0.1:" + std::to_string(port) + "/mcp");
    MCPClient client(transport);

    auto tools = client.list_tools();
    ASSERT_EQ(tools.size(), 2u);

    json resp = client.call_tool("echo", json{{"message", "over-http"}});
    EXPECT_EQ(resp["result"]["structuredContent"]["message"], "over-http");

    http.stop();
    t.join();
}

// ---------- E2E: stdio transport <-> real `serius mcp-serve` ----------

TEST(MCPServer, StdioEndToEndRealBinary)
{
    const auto exe = find_serius_exe();
    if (exe.empty()) {
        GTEST_SKIP() << "serius.exe not found (set SERIUS_EXE)";
    }

    // Quote the exe path: it may contain spaces.
    const std::string cmd = "\"" + exe.string() + "\" mcp-serve --audit mcp_e2e_audit.log";
    StdioTransport transport(cmd);
    MCPClient client(transport);

    auto tools = client.list_tools();
    ASSERT_FALSE(tools.empty());
    bool found_echo = false;
    for (const auto &tool : tools) {
        if (tool.name == "echo") { found_echo = true; }
    }
    EXPECT_TRUE(found_echo);

    json resp = client.call_tool("echo", json{{"message", "over-stdio"}});
    ASSERT_TRUE(resp.contains("result"));
    EXPECT_EQ(resp["result"]["structuredContent"]["message"], "over-stdio");
}
