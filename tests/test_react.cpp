#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "agent/action_parser.h"
#include "agent/context_manager.h"
#include "agent/model_interface.h"
#include "agent/react.h"

using serius::agent::ActionCall;
using serius::agent::ActionParser;
using serius::agent::ContextManager;
using serius::agent::Model;
using serius::agent::ReActEngine;
using serius::agent::SamplingParams;
using serius::agent::ToolResult;

// ---- Mock model that returns scripted responses in sequence ----
class ScriptedModel : public Model
{
   public:
    void add_response(std::string response) { responses_.push_back(std::move(response)); }

    std::string generate(const std::string &, const SamplingParams &) override
    {
        if (index_ < responses_.size()) {
            return responses_[index_++];
        }
        return "[ANSWER] fallback answer.\n";
    }

    size_t calls() const { return index_; }

   private:
    std::vector<std::string> responses_;
    size_t index_ = 0;
};

// ---- ActionParser tests ----

TEST(React, ActionParserExtractsAction)
{
    std::string text = "Some thought.\n[ACTION] {\"tool\":\"search\",\"args\":{\"query\":\"Tokyo\"}}\n";
    ActionCall call = ActionParser::parse(text);
    EXPECT_EQ(call.name, "search");
    EXPECT_NE(call.args_json.find("Tokyo"), std::string::npos);
}

TEST(React, ActionParserNoAction)
{
    ActionCall call = ActionParser::parse("Just a thought, no action.");
    EXPECT_TRUE(call.name.empty());
}

TEST(React, ActionParserHasActionAndAnswer)
{
    EXPECT_TRUE(ActionParser::has_action("blah [ACTION] x"));
    EXPECT_FALSE(ActionParser::has_action("no action here"));
    EXPECT_TRUE(ActionParser::has_answer("blah [ANSWER] done"));
    EXPECT_FALSE(ActionParser::has_answer("no answer here"));
}

TEST(React, ActionParserExtractsAnswer)
{
    std::string text = "thinking\n[ANSWER] The answer is 42.[/ANSWER]";
    EXPECT_EQ(ActionParser::parse_answer(text), " The answer is 42.");
}

// ---- ContextManager tests ----

TEST(React, ContextTruncateShort)
{
    std::string obs = "short";
    EXPECT_EQ(ContextManager::truncate_observation(obs, 100), "short");
}

TEST(React, ContextTruncateLong)
{
    std::string obs(500, 'x');
    std::string t = ContextManager::truncate_observation(obs, 100);
    EXPECT_EQ(t.size(), 100 + std::string("\n[...TRUNCATED...]").size());
    EXPECT_NE(t.find("[...TRUNCATED...]"), std::string::npos);
}

TEST(React, ContextManageShort)
{
    std::string ctx(50, 'a');
    EXPECT_EQ(ContextManager::manage_context(ctx, 10, 200), ctx);
}

TEST(React, ContextManageLong)
{
    std::string ctx(500, 'a');
    std::string m = ContextManager::manage_context(ctx, 10, 100);
    EXPECT_LE(m.size(), 100 + 10);  // head + marker + tail
    EXPECT_NE(m.find("[...]"), std::string::npos);
}

// ---- ReActEngine tests ----

TEST(React, FullLoopWithTool)
{
    ScriptedModel model;
    model.add_response("[THOUGHT] Devo cercare la capitale.\n[ACTION] {\"tool\":\"search\",\"args\":{\"query\":\"capitale Giappone\"}}\n");
    model.add_response("[THOUGHT] Trovato.\n[ANSWER] Tokyo, circa 13.96 milioni.\n");

    ReActEngine engine(model, 5);
    ToolResult results[] = {{"search", "Tokyo è la capitale del Giappone."}};

    std::string answer = engine.run("Quanti abitanti ha Tokyo?",
                                    "You are a helpful assistant.", results, 1);

    EXPECT_NE(answer.find("Tokyo"), std::string::npos);
    EXPECT_GE(model.calls(), 2u);
}

TEST(React, MaxStepsForcesAnswer)
{
    ScriptedModel model;
    // Always emit an action, never an answer → loop should exhaust max_steps
    // and then force a final answer.
    for (int i = 0; i < 10; ++i) {
        model.add_response("[THOUGHT] Still thinking.\n[ACTION] {\"tool\":\"x\",\"args\":{}}\n");
    }

    ReActEngine engine(model, 3);
    std::string answer = engine.run("Query", "Sys prompt", nullptr, 0);

    // Should have called model max_steps times for the loop + 1 final.
    EXPECT_GE(model.calls(), 3u);
    EXPECT_FALSE(answer.empty());
}

TEST(React, NoToolAvailable)
{
    ScriptedModel model;
    model.add_response("[THOUGHT] Need a tool.\n[ACTION] {\"tool\":\"unknown\",\"args\":{}}\n");
    model.add_response("[ANSWER] No tool available, giving up.\n");

    ReActEngine engine(model, 5);
    std::string answer = engine.run("Query", "Sys", nullptr, 0);

    EXPECT_FALSE(answer.empty());
}

TEST(React, EmptySystemPrompt)
{
    ScriptedModel model;
    model.add_response("[ANSWER] Hello.\n");
    ReActEngine engine(model, 3);
    std::string answer = engine.run("Hi", "");
    EXPECT_NE(answer.find("Hello"), std::string::npos);
}
