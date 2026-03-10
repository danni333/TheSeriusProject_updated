#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "inference/sampler.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::core;
using namespace serius::inference;

// ============================================================
// Sampler Tests (Fase 10)
// ============================================================

TEST(Sampler, GreedyDecoding)
{
    Tensor logits({5});
    float *d = logits.data();
    d[0] = 1.0f;
    d[1] = 5.0f;
    d[2] = 2.0f;
    d[3] = 3.0f;
    d[4] = 0.5f;

    int result = Sampler::greedy(logits);
    EXPECT_EQ(result, 1);
}

TEST(Sampler, TemperatureScaling)
{
    Tensor logits({4});
    float *d = logits.data();
    d[0] = 2.0f;
    d[1] = 4.0f;
    d[2] = 6.0f;
    d[3] = 8.0f;

    Sampler::apply_temperature(logits, 0.5f);

    // Values should be doubled (divided by 0.5)
    EXPECT_FLOAT_EQ(d[0], 4.0f);
    EXPECT_FLOAT_EQ(d[1], 8.0f);
    EXPECT_FLOAT_EQ(d[2], 12.0f);
    EXPECT_FLOAT_EQ(d[3], 16.0f);
}

TEST(Sampler, LowTemperatureIsNearGreedy)
{
    // With very low temperature, sampling should almost always pick the max
    Tensor logits({5});
    float *d = logits.data();
    d[0] = 1.0f;
    d[1] = 2.0f;
    d[2] = 10.0f;  // clear winner
    d[3] = 1.5f;
    d[4] = 0.5f;

    SamplerConfig config;
    config.temperature = 0.01f;
    config.top_k = 0;
    config.top_p = 1.0f;
    config.repetition_penalty = 1.0f;
    Sampler sampler(config);

    int counts[5] = {0};
    for (int i = 0; i < 100; i++) {
        sampler.set_seed(static_cast<unsigned>(i));
        int token = sampler.sample(logits);
        ASSERT_GE(token, 0);
        ASSERT_LT(token, 5);
        counts[token]++;
    }

    // Token 2 should win overwhelmingly
    EXPECT_GT(counts[2], 90);
}

TEST(Sampler, TopKFiltering)
{
    Tensor logits({6});
    float *d = logits.data();
    d[0] = 1.0f;
    d[1] = 5.0f;
    d[2] = 3.0f;
    d[3] = 4.0f;
    d[4] = 2.0f;
    d[5] = 6.0f;

    Sampler::apply_top_k(logits, 3);

    // Top 3 are indices 5(6.0), 1(5.0), 3(4.0)
    // Others should be -inf
    EXPECT_TRUE(std::isinf(d[0]) && d[0] < 0);
    EXPECT_FALSE(std::isinf(d[1]));  // kept
    EXPECT_TRUE(std::isinf(d[2]) && d[2] < 0);
    EXPECT_FALSE(std::isinf(d[3]));  // kept
    EXPECT_TRUE(std::isinf(d[4]) && d[4] < 0);
    EXPECT_FALSE(std::isinf(d[5]));  // kept
}

TEST(Sampler, TopKOnlySamplesFromTopK)
{
    Tensor logits({10});
    float *d = logits.data();
    for (int i = 0; i < 10; i++) d[i] = static_cast<float>(i);

    SamplerConfig config;
    config.temperature = 1.0f;
    config.top_k = 3;
    config.top_p = 1.0f;
    config.repetition_penalty = 1.0f;
    Sampler sampler(config);

    std::set<int> sampled_tokens;
    for (int i = 0; i < 200; i++) {
        sampler.set_seed(static_cast<unsigned>(i * 7));
        int token = sampler.sample(logits);
        sampled_tokens.insert(token);
    }

    // Should only sample from {7, 8, 9} (top-3)
    for (int t : sampled_tokens) {
        EXPECT_GE(t, 7) << "Sampled token " << t << " is outside top-3";
    }
}

TEST(Sampler, TopPFiltering)
{
    // Create logits where softmax gives clear probabilities
    Tensor logits({4});
    float *d = logits.data();
    d[0] = 10.0f;  // dominant
    d[1] = 5.0f;
    d[2] = 1.0f;
    d[3] = 0.0f;

    Sampler::apply_top_p(logits, 0.95f);

    // Token 0 has highest probability, and likely token 0+1 together exceed 0.95
    // Token 0 should definitely be kept
    EXPECT_FALSE(std::isinf(d[0]));
}

TEST(Sampler, TopPSumExceedsP)
{
    Tensor logits({5});
    float *d = logits.data();
    d[0] = 2.0f;
    d[1] = 2.0f;
    d[2] = 2.0f;
    d[3] = 2.0f;
    d[4] = 2.0f;

    // All equal -> softmax gives 0.2 each
    // top_p = 0.5 should keep about 3 tokens (cumsum: 0.2, 0.4, 0.6 > 0.5)
    Tensor work = logits.copy();
    Sampler::apply_top_p(work, 0.5f);

    int kept = 0;
    float *wd = work.data();
    for (int i = 0; i < 5; i++) {
        if (!std::isinf(wd[i])) kept++;
    }
    EXPECT_GE(kept, 2);
    EXPECT_LE(kept, 4);
}

TEST(Sampler, RepetitionPenalty)
{
    Tensor logits({5});
    float *d = logits.data();
    d[0] = 2.0f;
    d[1] = 3.0f;
    d[2] = -1.0f;
    d[3] = 4.0f;
    d[4] = 1.0f;

    std::vector<int> past = {1, 2};
    Sampler::apply_repetition_penalty(logits, past, 1.2f);

    // Positive logit (token 1): divided by penalty
    EXPECT_NEAR(d[1], 3.0f / 1.2f, 1e-5f);
    // Negative logit (token 2): multiplied by penalty
    EXPECT_NEAR(d[2], -1.0f * 1.2f, 1e-5f);
    // Untouched tokens
    EXPECT_FLOAT_EQ(d[0], 2.0f);
    EXPECT_FLOAT_EQ(d[3], 4.0f);
    EXPECT_FLOAT_EQ(d[4], 1.0f);
}

TEST(Sampler, RepetitionPenaltyReducesProbOfPast)
{
    Tensor logits({4});
    float *d = logits.data();
    d[0] = 5.0f;
    d[1] = 5.0f;
    d[2] = 5.0f;
    d[3] = 5.0f;

    // Without penalty, all tokens equally likely
    // With penalty on token 1, it should be sampled less
    SamplerConfig config;
    config.temperature = 1.0f;
    config.top_k = 0;
    config.top_p = 1.0f;
    config.repetition_penalty = 2.0f;
    Sampler sampler(config);

    int count_token1 = 0;
    std::vector<int> past = {1};
    for (int i = 0; i < 400; i++) {
        sampler.set_seed(static_cast<unsigned>(i * 13 + 7));
        int token = sampler.sample(logits, past);
        if (token == 1) count_token1++;
    }

    // Without penalty, expected ~100/400. With penalty of 2.0, should be much less.
    EXPECT_LT(count_token1, 150) << "Repetition penalty should reduce frequency of past tokens";
}

TEST(Sampler, Softmax)
{
    Tensor logits({3});
    logits.data()[0] = 1.0f;
    logits.data()[1] = 2.0f;
    logits.data()[2] = 3.0f;

    Tensor probs = Sampler::softmax(logits);

    // Sum should be 1.0
    float sum = 0;
    for (size_t i = 0; i < 3; i++) sum += probs.data()[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Should be monotonically increasing
    EXPECT_LT(probs.data()[0], probs.data()[1]);
    EXPECT_LT(probs.data()[1], probs.data()[2]);
}

TEST(Sampler, SoftmaxNumericalStability)
{
    // Large logits should not produce NaN/Inf
    Tensor logits({3});
    logits.data()[0] = 1000.0f;
    logits.data()[1] = 1001.0f;
    logits.data()[2] = 999.0f;

    Tensor probs = Sampler::softmax(logits);

    float sum = 0;
    for (size_t i = 0; i < 3; i++) {
        EXPECT_FALSE(std::isnan(probs.data()[i]));
        EXPECT_FALSE(std::isinf(probs.data()[i]));
        sum += probs.data()[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(Sampler, ZeroTemperatureIsGreedy)
{
    Tensor logits({4});
    logits.data()[0] = 1.0f;
    logits.data()[1] = 3.0f;
    logits.data()[2] = 2.0f;
    logits.data()[3] = 0.0f;

    SamplerConfig config;
    config.temperature = 0.0f;
    Sampler sampler(config);

    int token = sampler.sample(logits);
    EXPECT_EQ(token, 1);
}

TEST(Sampler, CategoricalSampleValidRange)
{
    Tensor probs({5});
    float *d = probs.data();
    d[0] = 0.1f;
    d[1] = 0.2f;
    d[2] = 0.3f;
    d[3] = 0.15f;
    d[4] = 0.25f;

    SamplerConfig config;
    Sampler sampler(config);

    for (int i = 0; i < 100; i++) {
        sampler.set_seed(static_cast<unsigned>(i));
        int token = sampler.categorical_sample(probs);
        EXPECT_GE(token, 0);
        EXPECT_LT(token, 5);
    }
}
