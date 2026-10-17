#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "reasoning/budget.h"
#include "reasoning/cot.h"
#include "reasoning/mcts.h"
#include "reasoning/verifier.h"

using serius::reasoning::ChainOfThought;
using serius::reasoning::ComputeBudget;
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
