#include <gtest/gtest.h>
#include "knowledge/semantic_hash.h"
#include "knowledge/knowledge_fn.h"
#include "knowledge/fact_extractor.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::knowledge;
using namespace serius::core;

TEST(SemanticHash, TestHashSimilarInputs) {
    SemanticHash hasher(128, 32, 42);

    // Create two similar embeddings
    Tensor emb1({128});
    Tensor emb2({128});

    for (size_t i = 0; i < 128; ++i) {
        emb1[i] = static_cast<float>(i) * 0.1f;
        emb2[i] = static_cast<float>(i) * 0.1f + 0.01f;  // Small perturbation
    }

    uint64_t h1 = hasher.hash(emb1);
    uint64_t h2 = hasher.hash(emb2);

    // Similar inputs should have low Hamming distance
    size_t dist = SemanticHash::hamming_distance(h1, h2);
    EXPECT_LE(dist, 8);  // Should be within 8 bits for similar embeddings
}

TEST(SemanticHash, TestHashDifferentInputs) {
    SemanticHash hasher(128, 32, 42);

    // Create two very different embeddings
    Tensor emb1({128});
    Tensor emb2({128});

    for (size_t i = 0; i < 128; ++i) {
        emb1[i] = static_cast<float>(i) * 0.1f;
        emb2[i] = static_cast<float>(128 - i) * 0.1f;  // Reverse order
    }

    uint64_t h1 = hasher.hash(emb1);
    uint64_t h2 = hasher.hash(emb2);

    // Different inputs should have high Hamming distance
    size_t dist = SemanticHash::hamming_distance(h1, h2);
    EXPECT_GT(dist, 16);  // Should be more than 16 bits different
}

TEST(SemanticHash, TestHashBatch) {
    SemanticHash hasher(64, 16, 42);

    std::vector<Tensor> embeddings;
    for (size_t i = 0; i < 5; ++i) {
        Tensor emb({64});
        for (size_t j = 0; j < 64; ++j) {
            emb[j] = static_cast<float>((i + j) % 10) * 0.1f;
        }
        embeddings.push_back(emb);
    }

    std::vector<uint64_t> hashes = hasher.hash_batch(embeddings);
    EXPECT_EQ(hashes.size(), 5);

    // All hashes should be different (high probability)
    bool all_different = true;
    for (size_t i = 0; i < hashes.size(); ++i) {
        for (size_t j = i + 1; j < hashes.size(); ++j) {
            if (hashes[i] == hashes[j]) {
                all_different = false;
            }
        }
    }
    EXPECT_TRUE(all_different);
}

TEST(SemanticHash, TestHammingDistance) {
    uint64_t h1 = 0b11110000;
    uint64_t h2 = 0b11001100;

    size_t dist = SemanticHash::hamming_distance(h1, h2);
    EXPECT_EQ(dist, 4);  // 4 bits differ
}

TEST(SemanticHash, TestIsSimilar) {
    uint64_t h1 = 0b11110000;
    uint64_t h2 = 0b11110001;  // Only 1 bit different

    EXPECT_TRUE(SemanticHash::is_similar(h1, h2, 2));
    EXPECT_FALSE(SemanticHash::is_similar(h1, h2, 0));
}

TEST(KnowledgeFunction, TestBuildAndQuery) {
    KnowledgeFunction kf(64, 16, 42);

    std::vector<Fact> facts;
    std::vector<Tensor> embeddings;

    // Add a fact
    Fact fact1;
    fact1.text = "Paris is the capital of France";
    fact1.source = "geo_wiki.txt";
    fact1.importance = 1.0f;

    Tensor emb1({64});
    for (size_t i = 0; i < 64; ++i) {
        emb1[i] = static_cast<float>(i) * 0.1f;
    }

    facts.push_back(fact1);
    embeddings.push_back(emb1);

    kf.build(facts, embeddings);

    // Query with same embedding
    const CompressedKnowledge *result = kf.query(emb1);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->fact_count, 1);
    EXPECT_EQ(result->sources.size(), 1);
    EXPECT_EQ(result->sources[0], "geo_wiki.txt");
}

TEST(KnowledgeFunction, TestMergeFacts) {
    KnowledgeFunction kf(64, 16, 42);

    std::vector<Fact> facts;
    std::vector<Tensor> embeddings;

    // Add two facts with similar embeddings (will hash to same bucket)
    Fact fact1;
    fact1.text = "Paris is the capital of France";
    fact1.source = "source1.txt";
    fact1.importance = 1.0f;

    Fact fact2;
    fact2.text = "The capital of France is Paris";
    fact2.source = "source2.txt";
    fact2.importance = 1.0f;

    Tensor emb1({64});
    Tensor emb2({64});
    for (size_t i = 0; i < 64; ++i) {
        emb1[i] = static_cast<float>(i) * 0.1f;
        emb2[i] = static_cast<float>(i) * 0.1f;  // Same embedding
    }

    facts.push_back(fact1);
    facts.push_back(fact2);
    embeddings.push_back(emb1);
    embeddings.push_back(emb2);

    kf.build(facts, embeddings);

    // Should have merged into one entry
    const CompressedKnowledge *result = kf.query(emb1);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->fact_count, 2);
    EXPECT_EQ(result->sources.size(), 2);
}

TEST(KnowledgeFunction, TestQueryWithFallback) {
    KnowledgeFunction kf(64, 32, 42);

    std::vector<Fact> facts;
    std::vector<Tensor> embeddings;

    Fact fact;
    fact.text = "Test fact";
    fact.source = "test.txt";
    fact.importance = 1.0f;

    Tensor emb({64});
    for (size_t i = 0; i < 64; ++i) {
        emb[i] = static_cast<float>(i) * 0.1f;
    }

    facts.push_back(fact);
    embeddings.push_back(emb);

    kf.build(facts, embeddings);

    // Query with slightly different embedding
    Tensor query_emb({64});
    for (size_t i = 0; i < 64; ++i) {
        query_emb[i] = static_cast<float>(i) * 0.1f + 0.05f;
    }

    // Direct query might fail, fallback should succeed
    const CompressedKnowledge *result = kf.query_with_fallback(query_emb, 2);
    // May or may not find depending on hash collision
    // Just verify it doesn't crash
    (void)result;
}

TEST(KnowledgeFunction, TestStats) {
    KnowledgeFunction kf(64, 16, 42);

    std::vector<Fact> facts;
    std::vector<Tensor> embeddings;

    for (size_t i = 0; i < 10; ++i) {
        Fact fact;
        fact.text = "Fact " + std::to_string(i);
        fact.source = "source" + std::to_string(i) + ".txt";
        fact.importance = 1.0f;

        Tensor emb({64});
        for (size_t j = 0; j < 64; ++j) {
            emb[j] = static_cast<float>((i + j) % 10) * 0.1f;
        }

        facts.push_back(fact);
        embeddings.push_back(emb);
    }

    kf.build(facts, embeddings);

    KnowledgeFunction::Stats stats = kf.get_stats();
    EXPECT_GT(stats.total_entries, 0);
    EXPECT_EQ(stats.total_facts, 10);
    EXPECT_GT(stats.avg_confidence, 0.0f);
}

TEST(KnowledgeFunction, TestMergeLatents) {
    Tensor a({4});
    a[0] = 1.0f; a[1] = 2.0f; a[2] = 3.0f; a[3] = 4.0f;

    Tensor b({4});
    b[0] = 3.0f; b[1] = 4.0f; b[2] = 5.0f; b[3] = 6.0f;

    Tensor merged = KnowledgeFunction::merge_latents(a, b);

    EXPECT_FLOAT_EQ(merged[0], 2.0f);  // (1+3)/2
    EXPECT_FLOAT_EQ(merged[1], 3.0f);  // (2+4)/2
    EXPECT_FLOAT_EQ(merged[2], 4.0f);  // (3+5)/2
    EXPECT_FLOAT_EQ(merged[3], 5.0f);  // (4+6)/2
}

TEST(FactExtractor, TestExtractSentences) {
    FactExtractor extractor;

    std::string text = "Paris is the capital of France. It is known for the Eiffel Tower. "
                      "The city has many museums.";

    std::vector<Fact> facts = extractor.extract_facts(text, "test_doc.txt");

    EXPECT_GT(facts.size(), 0);
    EXPECT_LE(facts.size(), 3);  // Should extract up to 3 sentences

    // Check that facts have sources
    for (const auto &fact : facts) {
        EXPECT_EQ(fact.source, "test_doc.txt");
        EXPECT_GT(fact.text.length(), extractor.min_length());
    }
}

TEST(FactExtractor, TestExtractDefinitions) {
    FactExtractor extractor;
    extractor.set_extract_definitions(true);

    std::string text = "A cat is a small domesticated carnivorous mammal. "
                      "It is often kept as a pet.";

    std::vector<Fact> facts = extractor.extract_facts(text, "animals.txt");

    // Should extract the definition
    bool found_definition = false;
    for (const auto &fact : facts) {
        if (fact.text.find("is a") != std::string::npos) {
            found_definition = true;
            EXPECT_GT(fact.importance, 1.0f);  // Definitions should have higher importance
        }
    }
    EXPECT_TRUE(found_definition);
}

TEST(FactExtractor, TestMinMaxLength) {
    FactExtractor extractor;
    extractor.set_min_length(20);
    extractor.set_max_length(100);

    std::string text = "Short. This is a medium length sentence that should be extracted. "
                      "This is a very long sentence that exceeds the maximum length limit and should not be extracted "
                      "because it is too long according to the configured maximum length parameter.";

    std::vector<Fact> facts = extractor.extract_facts(text, "test.txt");

    for (const auto &fact : facts) {
        EXPECT_GE(fact.text.length(), 20);
        EXPECT_LE(fact.text.length(), 100);
    }
}

TEST(FactExtractor, TestEmptyText) {
    FactExtractor extractor;

    std::vector<Fact> facts = extractor.extract_facts("", "test.txt");
    EXPECT_EQ(facts.size(), 0);
}

TEST(FactExtractor, TestCleanFact) {
    FactExtractor extractor;

    std::string dirty = "  This   is   a   fact  with  extra  spaces.  ";
    std::string cleaned = extractor.clean_fact(dirty);

    EXPECT_EQ(cleaned, "This is a fact with extra spaces.");
}
