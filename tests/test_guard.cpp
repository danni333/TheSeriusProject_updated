#include <gtest/gtest.h>

#include <functional>
#include <nlohmann/json.hpp>
#include <string>

#include "mcp/client.h"
#include "mcp/guard.h"
#include "mcp/scrubber.h"
#include "mcp/transport.h"

using serius::mcp::MCPClient;
using serius::mcp::MCPGuard;
using serius::mcp::RiskLevel;
using serius::mcp::Scrubber;
using serius::mcp::Transport;
using serius::mcp::json;

class MockTransport : public Transport
{
   public:
    std::vector<std::pair<std::string, json>> calls;
    json result;

    json send_request(const std::string &method,
                      const json &params = json::object()) override
    {
        calls.push_back({method, params});
        return result;
    }
};

class MockClient : public MCPClient
{
   public:
    MockClient(MockTransport &transport) : MCPClient(transport), transport_(transport) {}
    MockTransport &transport_;
};

// ---- Guard tests ----

TEST(Guard, SafeToolExecutedWithoutApproval)
{
    MockTransport transport;
    transport.result = json{{"result", "file content"}};
    MockClient client(transport);
    MCPGuard guard;

    json args = {{"path", "/tmp/a.txt"}};
    json out = guard.guarded_call(client, "read_file", args);

    EXPECT_EQ(out["result"], "file content");
    ASSERT_EQ(transport.calls.size(), 1u);
    EXPECT_EQ(transport.calls[0].first, "tools/call");
}

TEST(Guard, DangerousBlockedWithoutApproval)
{
    MockTransport transport;
    transport.result = json{{"result", "should not happen"}};
    MockClient client(transport);
    MCPGuard guard;

    json args = {{"cmd", "rm -rf /"}};
    json out = guard.guarded_call(client, "run_command", args);

    EXPECT_TRUE(out.contains("error"));
    EXPECT_EQ(out["error"], "Operazione rifiutata dall'utente");
    EXPECT_EQ(transport.calls.size(), 0u);  // no actual call
}

TEST(Guard, DangerousAllowedWithApproval)
{
    MockTransport transport;
    transport.result = json{{"result", "command output"}};
    MockClient client(transport);
    MCPGuard guard;

    std::function<bool(const std::string &, const json &)> approve =
        [](const std::string &, const json &) { return true; };
    json args = {{"cmd", "ls"}};
    json out = guard.guarded_call(client, "run_command", args, approve);

    EXPECT_EQ(out["result"], "command output");
    ASSERT_EQ(transport.calls.size(), 1u);
}

TEST(Guard, ModerateToolAudited)
{
    MockTransport transport;
    transport.result = json{{"result", "written"}};
    MockClient client(transport);
    MCPGuard guard;

    json args = {{"path", "/tmp/x.txt"}, {"content", "hi"}};
    json out = guard.guarded_call(client, "write_file", args);

    EXPECT_EQ(out["result"], "written");
    ASSERT_EQ(transport.calls.size(), 1u);
}

TEST(Guard, RiskLevelDefaults)
{
    EXPECT_EQ(MCPGuard::risk_level("read_file"), RiskLevel::SAFE);
    EXPECT_EQ(MCPGuard::risk_level("run_command"), RiskLevel::DANGEROUS);
    EXPECT_EQ(MCPGuard::risk_level("unknown_tool"), RiskLevel::MODERATE);
}

TEST(Guard, SetCustomRisk)
{
    MCPGuard guard;
    guard.set_risk("search", RiskLevel::DANGEROUS);
    EXPECT_EQ(MCPGuard::risk_level("search"), RiskLevel::DANGEROUS);
}

// ---- Scrubber tests ----

TEST(Scrubber, RedactsApiKey)
{
    Scrubber scrubber;
    std::string text = "api_key=sk-abcd1234efgh5678ijkl9012mnop3456qrst78";
    std::string out = scrubber.scrub(text);
    EXPECT_NE(out, text);
    EXPECT_EQ(out.find("api_key"), std::string::npos);
}

TEST(Scrubber, RedactsPassword)
{
    Scrubber scrubber;
    std::string text = "password=hunter2";
    std::string out = scrubber.scrub(text);
    EXPECT_NE(out, text);
    EXPECT_EQ(out.find("password"), std::string::npos);
}

TEST(Scrubber, RedactsGitHubToken)
{
    Scrubber scrubber;
    std::string text = "token ghp_abcdefghijklmnopqrstuvwxyz012345678";
    std::string out = scrubber.scrub(text);
    EXPECT_NE(out, text);
    EXPECT_EQ(out.find("ghp_abcdef"), std::string::npos);
}

TEST(Scrubber, CleanTextUnchanged)
{
    Scrubber scrubber;
    std::string text = "This is a normal sentence.";
    EXPECT_EQ(scrubber.scrub(text), text);
}

TEST(Scrubber, CustomPattern)
{
    Scrubber scrubber;
    scrubber.add_pattern(R"(SSN:\s*\d{3}-\d{2}-\d{4})");
    std::string text = "SSN: 123-45-6789";
    std::string out = scrubber.scrub(text);
    EXPECT_NE(out.find("[REDACTED]"), std::string::npos);
    EXPECT_EQ(out.find("123-45-6789"), std::string::npos);
}
