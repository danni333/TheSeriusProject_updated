#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "grounding/self_consistency.h"
#include "grounding/citation_engine.h"
#include "grounding/confidence.h"

using serius::grounding::cosine_similarity;
using serius::grounding::self_consistent_index;
using serius::grounding::majority_vote;
using serius::grounding::extract_citations;
using serius::grounding::support_score;
using serius::grounding::confidence_from_logprobs;

TEST(Grounding, CosineSimilarityIdentical)
{
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {1.0f, 0.0f, 0.0f};
    EXPECT_FLOAT_EQ(cosine_similarity(a, b), 1.0f);
}

TEST(Grounding, CosineSimilarityOrthogonal)
{
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    EXPECT_FLOAT_EQ(cosine_similarity(a, b), 0.0f);
}

TEST(Grounding, CosineSimilarityEmpty)
{
    std::vector<float> a;
    std::vector<float> b = {1.0f};
    EXPECT_FLOAT_EQ(cosine_similarity(a, b), 0.0f);
}

TEST(Grounding, SelfConsistentIndex)
{
    // Embedding 0 è simile a tutti gli altri; embedding 1 è un outlier.
    std::vector<std::vector<float>> embs = {
        {1.0f, 0.0f, 0.0f},
        {0.9f, 0.1f, 0.0f},
        {0.8f, 0.2f, 0.0f},
        {0.0f, 0.0f, 1.0f},  // outlier
    };
    const size_t idx = self_consistent_index(embs);
    EXPECT_LT(idx, embs.size());
    // L'indice scelto non dovrebbe essere l'outlier (indice 3)
    EXPECT_NE(idx, 3u);
}

TEST(Grounding, MajorityVote)
{
    std::vector<std::string> responses = {"Parigi", "Roma", "Parigi", "Parigi", "Londra"};
    EXPECT_EQ(majority_vote(responses), "Parigi");
}

TEST(Grounding, MajorityVoteSingle)
{
    std::vector<std::string> responses = {"Solo"};
    EXPECT_EQ(majority_vote(responses), "Solo");
}

TEST(Grounding, ExtractCitations)
{
    std::string text = "Secondo [1] il cielo è blu. Vedi anche [2] e [1].";
    const auto cits = extract_citations(text);
    ASSERT_EQ(cits.size(), 3u);
    EXPECT_EQ(cits[0].doc_id, 1);
    EXPECT_EQ(cits[0].token, "[1]");
    EXPECT_EQ(cits[1].doc_id, 2);
    EXPECT_EQ(cits[2].doc_id, 1);
}

TEST(Grounding, ExtractCitationsNone)
{
    const auto cits = extract_citations("nessuna citazione qui");
    EXPECT_TRUE(cits.empty());
}

TEST(Grounding, SupportScore)
{
    const std::string claim = "Parigi e la capitale della Francia";
    const std::string doc = "Parigi e la capitale della Francia e una citta europea";
    const float score = support_score(claim, doc);
    EXPECT_GT(score, 0.5f);
}

TEST(Grounding, SupportScoreNoOverlap)
{
    const float score = support_score("xyz abc", "qualunque altra cosa");
    EXPECT_FLOAT_EQ(score, 0.0f);
}

TEST(Grounding, ConfidenceHigh)
{
    const std::vector<float> log_probs = {-0.01f, -0.02f, -0.01f};
    const float conf = confidence_from_logprobs(log_probs);
    EXPECT_GT(conf, 0.9f);
    EXPECT_LE(conf, 1.0f);
}

TEST(Grounding, ConfidenceLow)
{
    const std::vector<float> log_probs = {-5.0f, -5.0f, -5.0f};
    const float conf = confidence_from_logprobs(log_probs);
    EXPECT_LT(conf, 0.01f);
}

TEST(Grounding, ConfidenceEmpty)
{
    EXPECT_FLOAT_EQ(confidence_from_logprobs({}), 0.0f);
}
