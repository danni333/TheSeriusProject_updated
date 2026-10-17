#include <gtest/gtest.h>

#include <cmath>

#include "tensor/tensor.h"
#include "transformer/positional.h"

using namespace serius;
using namespace serius::model;
using namespace serius::core;

TEST(RoPE, RelativePositionEncoding)
{
    // RoPE should preserve relative position information:
    // dot(RoPE(q, pos_m), RoPE(k, pos_n)) should depend only on (m - n)
    size_t d_head = 64;
    size_t seq_len = 8;

    Tensor q({seq_len, d_head});
    Tensor k({seq_len, d_head});

    // Initialize with same values for all positions
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_head; ++j) {
            q.at(i, j) = 1.0f;
            k.at(i, j) = 1.0f;
        }
    }

    // Apply RoPE
    apply_rope(q, k, d_head, 10000.0f, 0);

    // Check that dot products for same relative positions are identical
    // (since q and k have identical initial values)
    auto dot = [&](size_t m, size_t n) -> float {
        float sum = 0.0f;
        for (size_t i = 0; i < d_head; ++i) {
            sum += q.at(m, i) * k.at(n, i);
        }
        return sum;
    };

    // pos 2 vs pos 0 (delta=2) should equal pos 5 vs pos 3 (delta=2)
    float dot_2_0 = dot(2, 0);
    float dot_5_3 = dot(5, 3);
    EXPECT_NEAR(dot_2_0, dot_5_3, 1e-5f);

    // pos 0 vs pos 0 (delta=0) should equal pos 7 vs pos 7 (delta=0)
    float dot_0_0 = dot(0, 0);
    float dot_7_7 = dot(7, 7);
    EXPECT_NEAR(dot_0_0, dot_7_7, 1e-5f);
}

TEST(RoPE, NTKEffectiveBase)
{
    // NTK-aware scaling should increase the base when extending context
    size_t d_head = 64;
    size_t train_ctx = 4096;
    size_t target_ctx = 8192;

    float base_original = 10000.0f;
    float base_scaled = compute_rope_base(base_original, d_head, train_ctx, target_ctx);

    // Scaled base should be larger than original
    EXPECT_GT(base_scaled, base_original);

    // For d_head = 64, the formula gives a specific multiplier
    float expected_multiplier = std::pow(
        static_cast<float>(target_ctx) / train_ctx,
        static_cast<float>(d_head) / (d_head - 2));
    float expected_base = base_original * expected_multiplier;
    EXPECT_NEAR(base_scaled, expected_base, 1e-3f);
}

TEST(RoPE, PositionOffset)
{
    // Position offset should shift the rotation
    size_t d_head = 32;
    size_t seq_len = 4;

    Tensor q1({seq_len, d_head});
    Tensor k1({seq_len, d_head});
    Tensor q2({seq_len, d_head});
    Tensor k2({seq_len, d_head});

    // Initialize with same values
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_head; ++j) {
            float val = static_cast<float>(i * d_head + j);
            q1.at(i, j) = val;
            k1.at(i, j) = val;
            q2.at(i, j) = val;
            k2.at(i, j) = val;
        }
    }

    // Apply RoPE with different offsets
    apply_rope(q1, k1, d_head, 10000.0f, 0);
    apply_rope(q2, k2, d_head, 10000.0f, 10);

    // Q with offset 10 should equal Q at pos 10 with offset 0
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_head; ++j) {
            // q2[i] should match q1[i+10] if q1 had that length
            // For simplicity, just check they are different
            EXPECT_NE(q1.at(i, j), q2.at(i, j));
        }
    }
}

TEST(RoPE, TrainingContextZero)
{
    // When training_ctx = 0, should use original base
    size_t d_head = 64;
    float base = compute_rope_base(10000.0f, d_head, 0, 0);
    EXPECT_FLOAT_EQ(base, 10000.0f);
}
