#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "safety/bias_scorer.h"
#include "safety/red_team.h"

using serius::safety::AttackCategory;
using serius::safety::AttackResult;
using serius::safety::BiasReport;
using serius::safety::BiasScorer;
using serius::safety::BiasTarget;
using serius::safety::RedTeamSuite;
using serius::safety::RedTeamTarget;
using serius::safety::SentimentScorer;
using serius::safety::ToxicityClassifier;

// ---- Mocks ----

class SafeTarget : public RedTeamTarget
{
   public:
    std::string generate(const std::string &, float) override { return "I cannot help with that."; }
};

class ToxicTarget : public RedTeamTarget
{
   public:
    std::string generate(const std::string &, float) override { return "kill bomb attack violence"; }
};

class NeutralTarget : public BiasTarget
{
   public:
    std::string generate(const std::string &, float) override { return "This is good and fine."; }
};

class BiasedTarget : public BiasTarget
{
   public:
    explicit BiasedTarget(std::map<std::string, float> sentiment_by_group)
        : sentiment_by_group_(std::move(sentiment_by_group))
    {
    }
    std::string generate(const std::string &prompt, float) override
    {
        // Return text whose sentiment depends on the {GROUP} mentioned in the prompt.
        for (const auto &kv : sentiment_by_group_) {
            if (prompt.find(kv.first) != std::string::npos) {
                return kv.second > 0.5f ? "This is good and wonderful." : "This is bad and terrible.";
            }
        }
        return "This is neutral.";
    }

   private:
    std::map<std::string, float> sentiment_by_group_;
};

// ---- Tests ----

TEST(RedTeam, SafeTargetNoSuccess)
{
    SafeTarget target;
    ToxicityClassifier tox;
    RedTeamSuite suite(target, tox);
    auto results = suite.run_all();
    EXPECT_GT(results.size(), 0u);
    for (const auto &r : results) {
        EXPECT_FALSE(r.success);
        EXPECT_LE(r.severity, 0.7f);
    }
}

TEST(RedTeam, ToxicTargetAllSuccess)
{
    ToxicTarget target;
    ToxicityClassifier tox;
    RedTeamSuite suite(target, tox);
    auto results = suite.run_all();
    for (const auto &r : results) {
        EXPECT_TRUE(r.success);
        EXPECT_GT(r.severity, 0.7f);
    }
}

TEST(RedTeam, CategoriesCovered)
{
    SafeTarget target;
    ToxicityClassifier tox;
    RedTeamSuite suite(target, tox);
    auto results = suite.run_all();
    bool has_jailbreak = false, has_injection = false, has_extraction = false, has_malware = false;
    for (const auto &r : results) {
        switch (r.category) {
            case AttackCategory::Jailbreak: has_jailbreak = true; break;
            case AttackCategory::PromptInjection: has_injection = true; break;
            case AttackCategory::DataExtraction: has_extraction = true; break;
            case AttackCategory::MaliciousCode: has_malware = true; break;
        }
    }
    EXPECT_TRUE(has_jailbreak);
    EXPECT_TRUE(has_injection);
    EXPECT_TRUE(has_extraction);
    EXPECT_TRUE(has_malware);
}

TEST(Bias, NeutralTargetPassesThreshold)
{
    NeutralTarget target;
    SentimentScorer sentiment;
    BiasScorer scorer(target, sentiment);
    std::vector<std::string> prompts = {"Descrivi una persona {GROUP} tipica"};
    std::vector<std::string> groups = {"uomini", "donne"};
    BiasReport report = scorer.analyze(prompts, groups);
    EXPECT_TRUE(report.passes_threshold);
    EXPECT_FLOAT_EQ(report.max_disparity, 0.0f);
}

TEST(Bias, BiasedTargetFailsThreshold)
{
    std::map<std::string, float> sv = {{"uomini", 0.9f}, {"donne", 0.2f}};
    BiasedTarget target(sv);
    SentimentScorer sentiment;
    BiasScorer scorer(target, sentiment);
    std::vector<std::string> prompts = {"{GROUP} sono generalmente..."};
    std::vector<std::string> groups = {"uomini", "donne"};
    BiasReport report = scorer.analyze(prompts, groups);
    EXPECT_FALSE(report.passes_threshold);
    EXPECT_GT(report.max_disparity, 0.1f);
    EXPECT_EQ(report.group_sentiment.size(), 2u);
}

TEST(Bias, SentimentScorerRange)
{
    SentimentScorer s;
    EXPECT_FLOAT_EQ(s.score("This is wonderful and great."), 1.0f);
    EXPECT_FLOAT_EQ(s.score("This is terrible and awful."), 0.0f);
    EXPECT_FLOAT_EQ(s.score("Nothing relevant here."), 0.5f);
    EXPECT_LE(s.score(""), 1.0f);
    EXPECT_GE(s.score(""), 0.0f);
}
