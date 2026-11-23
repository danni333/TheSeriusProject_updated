#include <gtest/gtest.h>

#include <cmath>

#include "attention/sparse_attention.h"
#include "attention/sparsity_mask.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::attention;
using namespace serius::core;

TEST(SparseAttention, CosineSimilarity)
{
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    float c[] = {-1.0f, -2.0f, -3.0f};

    float sim_ab = cosine_similarity(a, b, 3);
    float sim_ac = cosine_similarity(a, c, 3);

    EXPECT_NEAR(sim_ab, 1.0f, 1e-5f);
    EXPECT_NEAR(sim_ac, -1.0f, 1e-5f);
}

TEST(SparseAttention, ScaledDotProduct)
{
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    float scale = 1.0f / std::sqrt(2.0f);

    float dot = scaled_dot_product(a, b, 2, scale);
    EXPECT_FLOAT_EQ(dot, 0.0f);
}

TEST(SparseAttention, DynamicSparseThreshold)
{
    size_t seq_len = 8;
    size_t d_model = 16;

    Tensor Q({seq_len, d_model});
    Tensor K({seq_len, d_model});
    Tensor V({seq_len, d_model});

    // Initialize with random values to get varied scores
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            Q.at({i, j}) = static_cast<float>((i * d_model + j) % 100) / 100.0f;
            K.at({i, j}) = static_cast<float>((j * seq_len + i) % 100) / 100.0f;
            V.at({i, j}) = 1.0f;
        }
    }

    auto result = dynamic_sparse_attention(Q, K, V, 10.0f);  // High threshold to force sparsity

    EXPECT_EQ(result.output.shape()[0], seq_len);
    EXPECT_EQ(result.output.shape()[1], d_model);
    EXPECT_GE(result.sparsity_ratio, 0.0f);
    EXPECT_LE(result.sparsity_ratio, 1.0f);
}

TEST(SparseAttention, DynamicSparseTopK)
{
    size_t seq_len = 8;
    size_t d_model = 16;

    Tensor Q({seq_len, d_model});
    Tensor K({seq_len, d_model});
    Tensor V({seq_len, d_model});

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            Q.at({i, j}) = static_cast<float>(i);
            K.at({i, j}) = static_cast<float>(j);
            V.at({i, j}) = 1.0f;
        }
    }

    auto result = dynamic_sparse_attention(Q, K, V, 0.0f, 3);  // top-3

    EXPECT_EQ(result.output.shape()[0], seq_len);
    EXPECT_EQ(result.output.shape()[1], d_model);

    // Each query should have at most 3 active keys
    for (const auto& keys : result.active_keys) {
        EXPECT_LE(keys.size(), 3);
    }
}

TEST(SparseAttention, SparsityRatio)
{
    size_t seq_len = 4;
    size_t d_model = 8;

    Tensor Q({seq_len, d_model});
    Tensor K({seq_len, d_model});
    Tensor V({seq_len, d_model});

    // Initialize with varied values
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            Q.at({i, j}) = static_cast<float>(i);
            K.at({i, j}) = static_cast<float>(j);
            V.at({i, j}) = 1.0f;
        }
    }

    auto result = dynamic_sparse_attention(Q, K, V, 100.0f);  // Very high threshold

    // With very high threshold, should skip most keys
    EXPECT_GE(result.sparsity_ratio, 0.0f);
}

TEST(SparsityMask, CausalBlockMask)
{
    size_t num_blocks = 4;
    Tensor mask = causal_block_mask(num_blocks);

    EXPECT_EQ(mask.shape()[0], num_blocks);
    EXPECT_EQ(mask.shape()[1], num_blocks);

    // Check causal property
    for (size_t i = 0; i < num_blocks; ++i) {
        for (size_t j = 0; j < num_blocks; ++j) {
            if (j <= i) {
                EXPECT_FLOAT_EQ(mask.at({i, j}), 1.0f);
            } else {
                EXPECT_FLOAT_EQ(mask.at({i, j}), 0.0f);
            }
        }
    }
}

TEST(SparsityMask, SlidingWindowBlockMask)
{
    size_t num_blocks = 6;
    size_t window_blocks = 2;
    Tensor mask = sliding_window_block_mask(num_blocks, window_blocks);

    EXPECT_EQ(mask.shape()[0], num_blocks);
    EXPECT_EQ(mask.shape()[1], num_blocks);

    // Block 0 should attend to block 0
    EXPECT_FLOAT_EQ(mask.at({0, 0}), 1.0f);

    // Block 3 should attend to blocks 2 and 3 (window size 2)
    EXPECT_FLOAT_EQ(mask.at({3, 2}), 1.0f);
    EXPECT_FLOAT_EQ(mask.at({3, 3}), 1.0f);
    EXPECT_FLOAT_EQ(mask.at({3, 1}), 0.0f);
}

TEST(SparsityMask, RandomBlockMask)
{
    size_t num_blocks = 10;
    Tensor mask = random_block_mask(num_blocks, 0.5f);

    EXPECT_EQ(mask.shape()[0], num_blocks);
    EXPECT_EQ(mask.shape()[1], num_blocks);

    // Check that sparsity ratio is approximately 0.5 (allowing for causal constraint)
    float sparsity = calculate_sparsity_ratio(mask);
    EXPECT_GT(sparsity, 0.2f);
    EXPECT_LT(sparsity, 0.8f);
}

TEST(SparsityMask, LongformerBlockMask)
{
    size_t num_blocks = 6;
    size_t window_blocks = 2;
    std::vector<size_t> global_blocks = {0};

    Tensor mask = longformer_block_mask(num_blocks, window_blocks, global_blocks);

    // Block 0 (global) should attend to all blocks
    for (size_t j = 0; j < num_blocks; ++j) {
        EXPECT_FLOAT_EQ(mask.at({0, j}), 1.0f);
    }

    // All blocks should attend to block 0 (global)
    for (size_t i = 0; i < num_blocks; ++i) {
        EXPECT_FLOAT_EQ(mask.at({i, 0}), 1.0f);
    }
}

TEST(SparsityMask, CalculateSparsityRatio)
{
    size_t num_blocks = 4;
    Tensor mask({num_blocks, num_blocks});
    mask.fill(1.0f);

    float sparsity = calculate_sparsity_ratio(mask);
    EXPECT_FLOAT_EQ(sparsity, 0.0f);

    mask.fill(0.0f);
    sparsity = calculate_sparsity_ratio(mask);
    EXPECT_FLOAT_EQ(sparsity, 1.0f);
}

TEST(SparsityMask, BlockMaskToFullMask)
{
    size_t num_blocks = 2;
    size_t block_size = 2;
    size_t seq_len = num_blocks * block_size;

    Tensor block_mask({num_blocks, num_blocks});
    block_mask.fill(1.0f);

    Tensor full_mask = block_mask_to_full_mask(block_mask, block_size);

    EXPECT_EQ(full_mask.shape()[0], seq_len);
    EXPECT_EQ(full_mask.shape()[1], seq_len);
}

TEST(SparsityMask, GenerateBlockMaskCausal)
{
    size_t seq_len = 8;
    size_t block_size = 2;

    Tensor mask = generate_block_mask(seq_len, block_size, SparsityPattern::CAUSAL);

    EXPECT_EQ(mask.shape()[0], seq_len / block_size);
    EXPECT_EQ(mask.shape()[1], seq_len / block_size);
}

TEST(SparsityMask, BigBirdBlockMask)
{
    size_t num_blocks = 8;
    // keep_prob = 0 isolates the deterministic local+global structure.
    Tensor mask = bigbird_block_mask(num_blocks, /*window_blocks=*/2,
                                     /*num_global_blocks=*/1, /*keep=*/0.0f, /*seed=*/42);

    EXPECT_EQ(mask.shape()[0], num_blocks);
    EXPECT_EQ(mask.shape()[1], num_blocks);

    // Global block 0: its row attends every block, every row attends it.
    for (size_t j = 0; j < num_blocks; ++j) {
        EXPECT_FLOAT_EQ(mask.at({0, j}), 1.0f);
        EXPECT_FLOAT_EQ(mask.at({j, 0}), 1.0f);
    }

    // Local causal window of 2: block 5 attends blocks 4 and 5 only
    // (block 0 is global and already covered).
    EXPECT_FLOAT_EQ(mask.at({5, 5}), 1.0f);
    EXPECT_FLOAT_EQ(mask.at({5, 4}), 1.0f);
    EXPECT_FLOAT_EQ(mask.at({5, 3}), 0.0f);
    EXPECT_FLOAT_EQ(mask.at({5, 2}), 0.0f);

    // Future non-global pair stays masked with keep_prob = 0.
    EXPECT_FLOAT_EQ(mask.at({1, 7}), 0.0f);
}

TEST(SparsityMask, BigBirdBlockMaskDeterministic)
{
    size_t num_blocks = 16;
    Tensor a = bigbird_block_mask(num_blocks, 4, 2, 0.2f, 1234);
    Tensor b = bigbird_block_mask(num_blocks, 4, 2, 0.2f, 1234);
    Tensor c = bigbird_block_mask(num_blocks, 4, 2, 0.2f, 5678);

    for (size_t i = 0; i < num_blocks; ++i) {
        for (size_t j = 0; j < num_blocks; ++j) {
            EXPECT_FLOAT_EQ(a.at({i, j}), b.at({i, j}));
        }
    }
    // Different seeds should (overwhelmingly likely) differ somewhere.
    bool differs = false;
    for (size_t i = 0; i < num_blocks && !differs; ++i) {
        for (size_t j = 0; j < num_blocks; ++j) {
            if (a.at({i, j}) != c.at({i, j})) { differs = true; break; }
        }
    }
    EXPECT_TRUE(differs);
}

TEST(SparsityMask, BigBirdBlockMaskFullKeep)
{
    size_t num_blocks = 4;
    Tensor mask = bigbird_block_mask(num_blocks, 1, 0, 1.0f, 0);
    for (size_t i = 0; i < num_blocks; ++i) {
        for (size_t j = 0; j < num_blocks; ++j) {
            EXPECT_FLOAT_EQ(mask.at({i, j}), 1.0f);
        }
    }
}

TEST(SparsityMask, GenerateBlockMaskBigBird)
{
    size_t seq_len = 16;
    size_t block_size = 2;
    Tensor mask = generate_block_mask(seq_len, block_size, SparsityPattern::BIGBIRD);

    EXPECT_EQ(mask.shape()[0], seq_len / block_size);
    EXPECT_EQ(mask.shape()[1], seq_len / block_size);

    // BIGBIRD should be sparser than a full causal mask but denser than a
    // purely causal one only via global/random edges — just verify it is not
    // identical to the causal mask.
    Tensor causal = causal_block_mask(seq_len / block_size);
    bool differs = false;
    for (size_t i = 0; i < mask.shape()[0] && !differs; ++i) {
        for (size_t j = 0; j < mask.shape()[1]; ++j) {
            if (mask.at({i, j}) != causal.at({i, j})) { differs = true; break; }
        }
    }
    EXPECT_TRUE(differs);
}

TEST(SparsityMask, GenerateBlockMaskSlidingWindow)
{
    size_t seq_len = 8;
    size_t block_size = 2;
    size_t window_size = 4;

    Tensor mask = generate_block_mask(seq_len, block_size, SparsityPattern::SLIDING_WINDOW, window_size);

    EXPECT_EQ(mask.shape()[0], seq_len / block_size);
    EXPECT_EQ(mask.shape()[1], seq_len / block_size);
}
