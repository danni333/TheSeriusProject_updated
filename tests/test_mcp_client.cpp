#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "mcp/client.h"
#include "mcp/transport.h"

using serius::mcp::MCPClient;
using serius::mcp::MCPTool;
using serius::mcp::Transport;
using serius::mcp::json;

// ---- Mock transport ----

class MockTransport : public Transport
{
   public:
    std::vector<std::pair<std::string, json>> sent_requests;
    json response_to_return;
    bool return_error = false;

    json send_request(const std::string &method,
                      const json &params = json::object()) override
    {
        sent_requests.push_back({method, params});
        if (return_error) {
            return json{{"jsonrpc", "2.0"}, {"id", nullptr},
                        {"error", {{"code", -32600}, {"message", "Invalid Request"}}}};
        }
        return response_to_return;
    }
};

// ---- Tests ----

TEST(MCPClient, JsonRpcRequestFormat)
{
    MockTransport transport;
    transport.response_to_return = json{{"jsonrpc", "2.0"}, {"id", 1},
                                        {"result", {{"tools", json::array()}}}};
    MCPClient client(transport);
    client.list_tools();

    ASSERT_EQ(transport.sent_requests.size(), 1u);
    EXPECT_EQ(transport.sent_requests[0].first, "tools/list");
    EXPECT_TRUE(transport.sent_requests[0].second.is_object());
}

TEST(MCPClient, ListToolsParsing)
{
    MockTransport transport;
    transport.response_to_return =
        json::parse(R"({"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"read_file","description":"Read a file","input_schema":{"type":"object"}}]}})");

    MCPClient client(transport);
    std::vector<serius::mcp::MCPTool> result;
    result = client.list_tools();

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].name, "read_file");
    EXPECT_EQ(result[0].description, "Read a file");
    EXPECT_TRUE(result[0].input_schema.contains("type"));
}

TEST(MCPClient, CallTool)
{
    MockTransport transport;
    transport.response_to_return =
        json::parse(R"({"jsonrpc":"2.0","id":2,"result":{"content":"file content here"}})");
    MCPClient client(transport);
    json args = {{"path", "/tmp/test.txt"}};
    json result = client.call_tool("read_file", args);

    ASSERT_EQ(transport.sent_requests.size(), 1u);
    EXPECT_EQ(transport.sent_requests[0].first, "tools/call");
    EXPECT_EQ(transport.sent_requests[0].second["name"], "read_file");
    EXPECT_EQ(transport.sent_requests[0].second["arguments"], args);
    EXPECT_EQ(result["result"]["content"], "file content here");
}

TEST(MCPClient, FormatToolsForPrompt)
{
    MCPTool t1;
    t1.name = "read_file";
    t1.description = "Read a file";
    t1.input_schema = json{{"type", "object"}};
    MCPTool t2;
    t2.name = "run_query";
    t2.description = "Run a SQL query";
    t2.input_schema = json{{"type", "object"}};
    std::vector<MCPTool> tools;
    tools.push_back(t1);
    tools.push_back(t2);

    std::string prompt = MCPClient::format_tools_for_prompt(tools);
    EXPECT_NE(prompt.find("read_file"), std::string::npos);
    EXPECT_NE(prompt.find("Read a file"), std::string::npos);
    EXPECT_NE(prompt.find("Available tools:"), std::string::npos);
    EXPECT_NE(prompt.find("run_query"), std::string::npos);
    EXPECT_NE(prompt.find("Run a SQL query"), std::string::npos);
}

TEST(MCPClient, ErrorResponsePropagated)
{
    MockTransport transport;
    transport.return_error = true;
    MCPClient client(transport);
    std::vector<serius::mcp::MCPTool> result = client.list_tools();
    EXPECT_TRUE(result.empty());  // error response → no tools parsed
}

TEST(MCPClient, EmptyToolsOnMissingResult)
{
    MockTransport transport;
    transport.response_to_return = json{{"jsonrpc", "2.0"}, {"id", 1}};  // no result
    MCPClient client(transport);
    std::vector<serius::mcp::MCPTool> result;
    result = client.list_tools();
    EXPECT_TRUE(result.empty());
}
