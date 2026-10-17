#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "federated/aggregator.h"
#include "federated/dp_noise.h"
#include "federated/user_adapter.h"

using serius::core::Tensor;
using serius::federated::add_dp_noise;
using serius::federated::compute_epsilon;
using serius::federated::compute_l2_norm;
using serius::federated::FederatedAggregator;
using serius::federated::UserAdapter;

// ---- DP noise tests ----

TEST(Federated, DpNoiseClipsNorm)
{
    Tensor delta({4, 4});
    delta.fill(3.0f);  // norm = 3 * 4 = 12, way above clip_norm

    float clip_norm = 2.0f;
    float sigma = 0.0f;  // no noise for deterministic check
    Tensor noisy = add_dp_noise(delta, clip_norm, sigma);

    float norm = compute_l2_norm(noisy);
    EXPECT_LE(norm, clip_norm + 1e-5f);
}

TEST(Federated, DpNoisePreservesSmallDelta)
{
    Tensor delta({2, 2});
    delta.fill(0.1f);  // norm = 0.2, below clip_norm

    float clip_norm = 1.0f;
    float sigma = 0.0f;
    Tensor noisy = add_dp_noise(delta, clip_norm, sigma);

    // With sigma=0 and norm below clip, result should equal input
    for (size_t i = 0; i < delta.size(); ++i) {
        EXPECT_FLOAT_EQ(noisy.data()[i], delta.data()[i]);
    }
}

TEST(Federated, DpNoiseAddsNoise)
{
    Tensor delta({10, 10});
    delta.fill(1.0f);

    float clip_norm = 5.0f;
    float sigma = 1.0f;
    Tensor noisy = add_dp_noise(delta, clip_norm, sigma);

    // Result should differ from input (noise added)
    bool differs = false;
    for (size_t i = 0; i < delta.size(); ++i) {
        if (noisy.data()[i] != delta.data()[i]) {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs);
}

// ---- Aggregation tests ----

TEST(Federated, AggregationAveragesUpdates)
{
    FederatedAggregator agg;

    Tensor delta({3, 3});
    delta.fill(2.0f);

    // Add 5 identical updates
    for (size_t i = 0; i < 5; ++i) {
        agg.add_update(i, delta);
    }

    Tensor avg = agg.aggregate();
    EXPECT_EQ(avg.shape(), delta.shape());
    for (size_t i = 0; i < avg.size(); ++i) {
        EXPECT_FLOAT_EQ(avg.data()[i], 2.0f);
    }
}

TEST(Federated, AggregationEmptyReturnsZeroTensor)
{
    FederatedAggregator agg;
    Tensor result = agg.aggregate();
    EXPECT_EQ(result.size(), 0u);
}

TEST(Federated, AggregationDifferentValues)
{
    FederatedAggregator agg;

    Tensor d1({2, 2});
    d1.fill(1.0f);
    Tensor d2({2, 2});
    d2.fill(3.0f);
    Tensor d3({2, 2});
    d3.fill(5.0f);

    agg.add_update(1, d1);
    agg.add_update(2, d2);
    agg.add_update(3, d3);

    Tensor avg = agg.aggregate();
    EXPECT_FLOAT_EQ(avg.data()[0], 3.0f);  // (1+3+5)/3 = 3
}

TEST(Federated, AggregationReset)
{
    FederatedAggregator agg;
    Tensor delta({1, 1});
    delta.fill(7.0f);
    agg.add_update(1, delta);
    EXPECT_EQ(agg.num_updates(), 1u);

    agg.reset();
    EXPECT_EQ(agg.num_updates(), 0u);
    Tensor result = agg.aggregate();
    EXPECT_EQ(result.size(), 0u);
}

// ---- Privacy budget tests ----

TEST(Federated, PrivacyBudgetHigherSigmaLowerEpsilon)
{
    float eps_low_sigma = compute_epsilon(0.1f, 10, 1e-5f);
    float eps_high_sigma = compute_epsilon(10.0f, 10, 1e-5f);
    EXPECT_LT(eps_high_sigma, eps_low_sigma);
}

TEST(Federated, PrivacyBudgetMoreStepsHigherEpsilon)
{
    float eps_few_steps = compute_epsilon(1.0f, 1, 1e-5f);
    float eps_many_steps = compute_epsilon(1.0f, 100, 1e-5f);
    EXPECT_LT(eps_few_steps, eps_many_steps);
}

TEST(Federated, PrivacyBudgetPositive)
{
    float eps = compute_epsilon(1.0f, 10, 1e-5f);
    EXPECT_GT(eps, 0.0f);
}

// ---- UserAdapter tests ----

TEST(Federated, UserAdapterConstruction)
{
    UserAdapter adapter("user_42");
    EXPECT_EQ(adapter.user_id(), "user_42");
    EXPECT_EQ(adapter.num_layers(), 0u);
}

TEST(Federated, UserAdapterLocalPath)
{
    UserAdapter adapter("user_99");
    adapter.set_local_path("/tmp/user_99_adapter.bin");
    EXPECT_EQ(adapter.local_path(), "/tmp/user_99_adapter.bin");
}
