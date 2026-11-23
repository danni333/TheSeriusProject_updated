#include <gtest/gtest.h>

#include <cmath>
#include <initializer_list>
#include <string>
#include <vector>

#include "reasoning/budget.h"
#include "reasoning/cot.h"
#include "reasoning/mcts.h"
#include "reasoning/verifier.h"

using serius::reasoning::ChainOfThought;
using serius::reasoning::ComputeBudget;
using serius::reasoning::EmbeddingVerifier;
using serius::reasoning::KeywordVerifier;
using serius::reasoning::MCTSModel;
using serius::reasoning::MCTSReasoner;
using serius::reasoning::Model;
using serius::reasoning::ThinkingResult;
using serius::reasoning::Verifier;

// ---- Mocks ----

class ScriptedModel : public Model
{
   public:
    void add_response(std::string response) { responses_.push_back(std::move(response)); }

    std::string generate(const std::string &, int, const std::string &) override
    {
        if (index_ < responses_.size()) {
            return responses_[index_++];
        }
        return "default";
    }

    size_t calls() const { return index_; }

   private:
    std::vector<std::string> responses_;
    size_t index_ = 0;
};

class ScriptedMCTSModel : public MCTSModel
{
   public:
    void add_response(std::string response) { responses_.push_back(std::move(response)); }

    std::string generate(const std::string &, int, float) override
    {
        if (index_ < responses_.size()) {
            return responses_[index_++];
        }
        return "step";
    }

   private:
    std::vector<std::string> responses_;
    size_t index_ = 0;
};

// ---- CoT tests ----

TEST(Reasoning, CoTProducesThinkingAndAnswer)
{
    ScriptedModel model;
    model.add_response("First, I analyze the problem.\nThen, I compute the result.");
    model.add_response("The answer is 42.");

    ChainOfThought cot(model);
    ThinkingResult r = cot.generate_with_thinking("What is 6 times 7?");

    EXPECT_FALSE(r.thinking.empty());
    EXPECT_FALSE(r.answer.empty());
    EXPECT_NE(r.answer.find("42"), std::string::npos);
    EXPECT_GE(model.calls(), 2u);
}

TEST(Reasoning, CoTThinkingIsSeparate)
{
    ScriptedModel model;
    model.add_response("thinking content here");
    model.add_response("final answer here");

    ChainOfThought cot(model);
    ThinkingResult r = cot.generate_with_thinking("Q");

    EXPECT_NE(r.thinking, r.answer);
}

// ---- Budget tests ----

TEST(Reasoning, BudgetMathQueryIsComplex)
{
    int b = ComputeBudget::estimate_budget("Calculate 15 percent of 200.");
    EXPECT_EQ(b, ComputeBudget::kComplex);
}

TEST(Reasoning, BudgetCodeQueryIsComplex)
{
    int b = ComputeBudget::estimate_budget("Write a python function to sort a list.");
    EXPECT_EQ(b, ComputeBudget::kComplex);
}

TEST(Reasoning, BudgetSimpleQueryIsSimple)
{
    int b = ComputeBudget::estimate_budget("What is the capital of France?");
    EXPECT_EQ(b, ComputeBudget::kSimple);
}

// ---- Verifier tests ----

TEST(Reasoning, VerifierKeywordPresent)
{
    KeywordVerifier v("Paris", {"error"});
    EXPECT_GT(v.score("The answer is Paris."), 0.5f);
}

TEST(Reasoning, VerifierKeywordAbsent)
{
    KeywordVerifier v("Paris", {"error"});
    EXPECT_FLOAT_EQ(v.score("The answer is London."), 0.0f);
}

TEST(Reasoning, VerifierFailureKeywordsReduceScore)
{
    KeywordVerifier v("Paris", {"error"});
    float with_error = v.score("The answer is Paris. error occurred.");
    float without_error = v.score("The answer is Paris.");
    EXPECT_GT(without_error, with_error);
}

namespace {

// Deterministic test embedder: counts topic keyword occurrences into
// dedicated dimensions, so semantic similarity tracks topic overlap.
std::vector<float> topic_embed(const std::string &text)
{
    std::string low = text;
    for (auto &c : low) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    auto count = [&](std::initializer_list<const char *> kws) {
        int n = 0;
        for (auto k : kws) {
            size_t pos = 0;
            const std::string key(k);
            while ((pos = low.find(key, pos)) != std::string::npos) {
                ++n;
                pos += key.size();
            }
        }
        return n;
    };
    std::vector<float> v(3, 0.0f);
    v[0] = static_cast<float>(count({"paris", "france", "eiffel", "capital"}));
    v[1] = static_cast<float>(count({"quantum", "quark", "chromodynamics", "physics"}));
    // Dimension 2: words outside the topic groups (background).
    v[2] = 1.0f;
    float n = 0.0f;
    for (float x : v) {
        n += x * x;
    }
    n = std::sqrt(n);
    for (auto &x : v) {
        x /= n;
    }
    return v;
}

}  // namespace

TEST(Reasoning, EmbeddingVerifierRanksSimilarHigher)
{
    EmbeddingVerifier v(topic_embed,
                        "The capital of France is Paris near the Eiffel Tower");

    const float similar = v.score("Paris is the capital city of France, "
                                  "home to the Eiffel Tower");
    const float unrelated = v.score("Quantum chromodynamics describes "
                                    "quark interactions");

    EXPECT_GT(similar, unrelated);
    EXPECT_GT(similar, 0.5f);
}

TEST(Reasoning, EmbeddingVerifierFailureKeywordsStillPenalize)
{
    EmbeddingVerifier v(topic_embed,
                        "The capital of France is Paris",
                        {"error"});

    const float clean = v.score("Paris is the capital of France");
    const float with_error = v.score("Paris is the capital of France. error");
    EXPECT_GT(clean, with_error);
}

TEST(Reasoning, EmbeddingVerifierLexicalFallbackWithoutEmbedder)
{
    EmbeddingVerifier v(nullptr, "The capital of France is Paris");

    const float similar = v.score("capital France Paris");
    const float unrelated = v.score("quantum chromodynamics quark");
    EXPECT_GT(similar, unrelated);
}

// ---- MCTS tests ----

TEST(Reasoning, MCTSExploresAndFindsBestPath)
{
    ScriptedMCTSModel model;
    // Each expansion returns 3 steps; the verifier rewards paths containing "Paris".
    for (int i = 0; i < 20; ++i) {
        model.add_response(" go to Paris.");
    }

    KeywordVerifier verifier("Paris", {});
    MCTSReasoner reasoner(model, verifier, 5);

    std::string result = reasoner.search("Plan a trip");

    EXPECT_FALSE(result.empty());
    // The best path should contain "Paris" because it's rewarded.
    EXPECT_NE(result.find("Paris"), std::string::npos);
}

TEST(Reasoning, MCTSRunsWithoutCrash)
{
    ScriptedMCTSModel model;
    model.add_response("a");
    model.add_response("b");

    KeywordVerifier verifier("x", {});
    MCTSReasoner reasoner(model, verifier, 2);

    std::string result = reasoner.search("problem");
    EXPECT_FALSE(result.empty());
}

// ---- agent::Model adapters ----

#include "reasoning/agent_model_adapter.h"
#include "agent/model_interface.h"

namespace {

class RecordingAgentModel : public serius::agent::Model
{
   public:
    std::string generate(const std::string &prompt,
                         const serius::agent::SamplingParams &params) override
    {
        last_prompt = prompt;
        last_params = params;
        calls++;
        if (responses.empty()) { return "ok"; }
        size_t idx = static_cast<size_t>(calls - 1);
        if (idx >= responses.size()) { idx = responses.size() - 1; }
        return responses[idx];
    }

    std::string last_prompt;
    serius::agent::SamplingParams last_params;
    std::vector<std::string> responses;
    int calls = 0;
};

}  // namespace

TEST(Reasoning, CoTAdapterMapsStopTokenAndMaxTokens)
{
    RecordingAgentModel agent_model;
    agent_model.responses = {"thinking...", "final answer"};
    serius::reasoning::AgentReasoningModel cot_model(agent_model);

    ChainOfThought cot(cot_model, 128);
    ThinkingResult r = cot.generate_with_thinking("Q");

    EXPECT_FALSE(r.answer.empty());
    ASSERT_EQ(agent_model.calls, 2);
    // Second call (answer phase) must carry the "</answer>" stop token
    // and the CoT max_tokens only applied to the thinking phase.
    EXPECT_EQ(agent_model.last_params.max_tokens, 1024);
    ASSERT_EQ(agent_model.last_params.stop_tokens.size(), 1u);
    EXPECT_EQ(agent_model.last_params.stop_tokens[0], "</answer>");
}

TEST(Reasoning, MCTSAdapterForwardsTemperature)
{
    RecordingAgentModel agent_model;
    serius::reasoning::AgentMCTSModel mcts_model(agent_model);

    (void)mcts_model.generate("prompt", 64, 0.3f);

    EXPECT_EQ(agent_model.last_params.max_tokens, 64);
    EXPECT_FLOAT_EQ(agent_model.last_params.temperature, 0.3f);
    EXPECT_EQ(agent_model.last_prompt, "prompt");
}
