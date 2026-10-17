#include <gtest/gtest.h>

#include <limits>

#include "attention/sliding_window.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::attention;
using namespace serius::core;

TEST(SlidingWindow, ValidWindow)
{
    EXPECT_TRUE(is_valid_window(10, 5));
    EXPECT_TRUE(is_valid_window(10, 10));
    EXPECT_FALSE(is_valid_window(10, 0));
    EXPECT_FALSE(is_valid_window(10, 11));
}

TEST(SlidingWindow, MaskShape)
{
    size_t seq_len = 8;
    size_t window_size = 3;
    Tensor mask = sliding_window_mask(seq_len, window_size);

    EXPECT_EQ(mask.ndim(), 2);
    EXPECT_EQ(mask.shape()[0], seq_len);
    EXPECT_EQ(mask.shape()[1], seq_len);
}

TEST(SlidingWindow, MaskCorrectness)
{
    size_t seq_len = 6;
    size_t window_size = 3;
    Tensor mask = sliding_window_mask(seq_len, window_size);

    const float NEG_INF = -std::numeric_limits<float>::infinity();

    // Position 0: can attend to [0]
    EXPECT_FLOAT_EQ(mask.at(0, 0), 0.0f);
    for (size_t j = 1; j < seq_len; ++j) {
        EXPECT_FLOAT_EQ(mask.at(0, j), NEG_INF);
    }

    // Position 2: can attend to [0, 1, 2] (window = 3)
    EXPECT_FLOAT_EQ(mask.at(2, 0), 0.0f);
    EXPECT_FLOAT_EQ(mask.at(2, 1), 0.0f);
    EXPECT_FLOAT_EQ(mask.at(2, 2), 0.0f);
    EXPECT_FLOAT_EQ(mask.at(2, 3), NEG_INF);
    EXPECT_FLOAT_EQ(mask.at(2, 4), NEG_INF);
    EXPECT_FLOAT_EQ(mask.at(2, 5), NEG_INF);

    // Position 5: can attend to [3, 4, 5] (window = 3)
    EXPECT_FLOAT_EQ(mask.at(5, 0), NEG_INF);
    EXPECT_FLOAT_EQ(mask.at(5, 1), NEG_INF);
    EXPECT_FLOAT_EQ(mask.at(5, 2), NEG_INF);
    EXPECT_FLOAT_EQ(mask.at(5, 3), 0.0f);
    EXPECT_FLOAT_EQ(mask.at(5, 4), 0.0f);
    EXPECT_FLOAT_EQ(mask.at(5, 5), 0.0f);
}

TEST(SlidingWindow, InvalidWindowFallback)
{
    size_t seq_len = 5;
    size_t window_size = 10;  // Invalid: larger than seq_len
    Tensor mask = sliding_window_mask(seq_len, window_size);

    // Should fallback to full attention (causal)
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            EXPECT_FLOAT_EQ(mask.at(i, j), 0.0f);
        }
        for (size_t j = i + 1; j < seq_len; ++j) {
            EXPECT_FLOAT_EQ(mask.at(i, j), -std::numeric_limits<float>::infinity());
        }
    }
}

TEST(SlidingWindow, GlobalLocalMask)
{
    size_t seq_len = 6;
    size_t window_size = 2;
    std::vector<size_t> global_indices = {0};  // First token is global

    Tensor mask = global_local_mask(seq_len, window_size, global_indices);

    const float NEG_INF = -std::numeric_limits<float>::infinity();

    // Global token (0) can attend to everything
    for (size_t j = 0; j < seq_len; ++j) {
        EXPECT_FLOAT_EQ(mask.at(0, j), 0.0f);
    }

    // All tokens can attend to global token (0)
    for (size_t i = 0; i < seq_len; ++i) {
        EXPECT_FLOAT_EQ(mask.at(i, 0), 0.0f);
    }

    // Position 3 (non-global): can attend to [0] (global) + [2, 3] (window)
    EXPECT_FLOAT_EQ(mask.at(3, 0), 0.0f);  // global
    EXPECT_FLOAT_EQ(mask.at(3, 1), NEG_INF);  // outside window
    EXPECT_FLOAT_EQ(mask.at(3, 2), 0.0f);  // in window
    EXPECT_FLOAT_EQ(mask.at(3, 3), 0.0f);  // in window
    EXPECT_FLOAT_EQ(mask.at(3, 4), NEG_INF);  // outside window
    EXPECT_FLOAT_EQ(mask.at(3, 5), NEG_INF);  // outside window
}

TEST(SlidingWindow, GlobalLocalMultipleGlobal)
{
    size_t seq_len = 8;
    size_t window_size = 2;
    std::vector<size_t> global_indices = {0, 4};  // Two global tokens

    Tensor mask = global_local_mask(seq_len, window_size, global_indices);

    const float NEG_INF = -std::numeric_limits<float>::infinity();

    // Both global tokens can attend to everything
    for (size_t j = 0; j < seq_len; ++j) {
        EXPECT_FLOAT_EQ(mask.at(0, j), 0.0f);
        EXPECT_FLOAT_EQ(mask.at(4, j), 0.0f);
    }

    // All tokens can attend to both global tokens
    for (size_t i = 0; i < seq_len; ++i) {
        EXPECT_FLOAT_EQ(mask.at(i, 0), 0.0f);
        EXPECT_FLOAT_EQ(mask.at(i, 4), 0.0f);
    }

    // Position 6 (non-global): can attend to [0, 4] (global) + [5, 6] (window)
    EXPECT_FLOAT_EQ(mask.at(6, 0), 0.0f);  // global
    EXPECT_FLOAT_EQ(mask.at(6, 4), 0.0f);  // global
    EXPECT_FLOAT_EQ(mask.at(6, 5), 0.0f);  // in window
    EXPECT_FLOAT_EQ(mask.at(6, 6), 0.0f);  // in window
    EXPECT_FLOAT_EQ(mask.at(6, 7), NEG_INF);  // outside window
}
