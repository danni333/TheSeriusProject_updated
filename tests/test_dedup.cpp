#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <vector>

#include "data/minhash.h"
#include "data/lsh.h"
#include "data/dedup.h"
#include "data/synthetic.h"

using serius::data::Document;
using serius::data::DedupPipeline;
using serius::data::DedupResult;
using serius::data::LSHIndex;
using serius::data::minhash_jaccard;
using serius::data::minhash_signature;
using serius::data::hash64;
using serius::data::HeuristicJudge;
using serius::data::SyntheticGenerator;
using serius::data::SyntheticSample;

TEST(DataCuration, Hash64Deterministic)
{
    EXPECT_EQ(hash64("hello"), hash64("hello"));
    EXPECT_NE(hash64("hello"), hash64("world"));
}

TEST(DataCuration, MinHashSimilarDocsSimilarSignatures)
{
    std::string doc_a = "the quick brown fox jumps over the lazy dog";
    std::string doc_b = "the quick brown fox jumps over the lazy dog.";
    std::string doc_c = "completely different unrelated text about quantum physics";

    auto sig_a = minhash_signature(doc_a, 128, 5);
    auto sig_b = minhash_signature(doc_b, 128, 5);
    auto sig_c = minhash_signature(doc_c, 128, 5);

    float jac_ab = minhash_jaccard(sig_a, sig_b);
    float jac_ac = minhash_jaccard(sig_a, sig_c);

    // Similar docs should have higher Jaccard than dissimilar ones.
    EXPECT_GT(jac_ab, jac_ac);
}

TEST(DataCuration, MinHashIdenticalDocsJaccardOne)
{
    std::string doc = "the quick brown fox jumps over the lazy dog";
    auto sig_a = minhash_signature(doc, 128, 5);
    auto sig_b = minhash_signature(doc, 128, 5);
    EXPECT_FLOAT_EQ(minhash_jaccard(sig_a, sig_b), 1.0f);
}

TEST(DataCuration, LSHFindsCandidates)
{
    LSHIndex lsh(8, 8);  // 64 rows total
    std::vector<uint32_t> sig_a(64, 100);
    std::vector<uint32_t> sig_b(64, 100);  // identical -> same band hashes
    std::vector<uint32_t> sig_c(64, 200);  // different

    lsh.insert(1, sig_a);
    lsh.insert(2, sig_b);
    lsh.insert(3, sig_c);

    auto candidates = lsh.find_candidates();
    // (1,2) should be a candidate; (1,3) and (2,3) should not.
    EXPECT_EQ(candidates.size(), 1u);
    EXPECT_EQ(*candidates.begin(), std::make_pair(1, 2));
}

TEST(DataCuration, DedupRemovesDuplicatesPreservesUniques)
{
    std::vector<Document> docs = {
        {0, "the quick brown fox jumps over the lazy dog"},
        {1, "the quick brown fox jumps over the lazy dog."},   // near-duplicate of 0
        {2, "completely different text about quantum physics"}, // unique
        {3, "another unique document about distributed systems"}, // unique
    };

    DedupPipeline pipeline(128, 16, 8, 0.8f, 5);
    DedupResult result = pipeline.run(docs);

    // Doc 0 and doc 2 and doc 3 should be kept; doc 1 removed as duplicate of 0.
    EXPECT_TRUE(result.kept_ids.count(0));
    EXPECT_TRUE(result.kept_ids.count(2));
    EXPECT_TRUE(result.kept_ids.count(3));
    EXPECT_TRUE(result.removed_ids.count(1));
    EXPECT_EQ(result.kept_ids.size(), 3u);
    EXPECT_EQ(result.removed_ids.size(), 1u);
    EXPECT_GE(result.duplicate_pairs, 1u);
}

TEST(DataCuration, DedupAllUniquesKept)
{
    std::vector<Document> docs = {
        {0, "alpha beta gamma delta epsilon"},
        {1, "zeta eta theta iota kappa lambda"},
        {2, "mu nu xi omicron pi rho sigma"},
    };

    DedupPipeline pipeline(64, 8, 8, 0.8f, 4);
    DedupResult result = pipeline.run(docs);

    EXPECT_EQ(result.kept_ids.size(), 3u);
    EXPECT_EQ(result.removed_ids.size(), 0u);
}

TEST(DataCuration, HeuristicJudgeScores)
{
    HeuristicJudge judge;
    // Short output -> low score.
    float low = judge.score("hi", "x");
    // Longer, well-formed output -> higher score.
    float high = judge.score("hi", "This is a complete sentence.");
    EXPECT_GT(high, low);
    EXPECT_GE(high, 0.0f);
    EXPECT_LE(high, 1.0f);
    EXPECT_LE(low, 1.0f);
}

TEST(DataCuration, SyntheticGeneratorAcceptsQuality)
{
    HeuristicJudge judge;
    int instruction_call = 0;
    auto gen_instruction = [&]() {
        return "Instruction " + std::to_string(++instruction_call);
    };
    auto gen_output = [](const std::string &inst) {
        return "Output for " + inst + " with some meaningful content here.";
    };

    SyntheticGenerator generator(gen_instruction, gen_output, judge, 0.7f);
    auto samples = generator.generate(5);

    ASSERT_EQ(samples.size(), 5u);
    for (const auto &s : samples) {
        EXPECT_GE(s.quality, 0.7f);
        EXPECT_FALSE(s.instruction.empty());
        EXPECT_FALSE(s.output.empty());
    }
}

TEST(DataCuration, SyntheticGeneratorRespectsThreshold)
{
    HeuristicJudge judge;
    auto gen_instruction = []() { return "Instr"; };
    // Outputs that always score 0 (single char, no punctuation).
    auto gen_output = [](const std::string &) { return "x"; };

    SyntheticGenerator generator(gen_instruction, gen_output, judge, 0.9f);
    auto samples = generator.generate(3);
    // Should not accept any low-quality samples within attempt budget.
    EXPECT_TRUE(samples.empty());
}
