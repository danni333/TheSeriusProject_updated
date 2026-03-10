#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "attention/flash_attn.h"
#include "tensor/tensor.h"

namespace serius {
namespace test {

using core::Tensor;
using model::attention::flash_attention;
using model::attention::flash_attention_forward;
using model::attention::FlashAttentionConfig;

namespace {
constexpr int BATCH_SIZE = 1;
constexpr int NUM_HEADS = 2;
constexpr int SEQ_LEN_Q = 128;
constexpr int SEQ_LEN_K = 128;
constexpr int HEAD_DIM = 64;

constexpr float SOFTMAX_SCALE_64 = 0.125F;  // 1/sqrt(64)
constexpr float TOLERANCE = 1e-3F;
}  // namespace

TEST(FlashAttentionTest, ForwardParity)
{
    // Skip if no CUDA
#ifndef SERIUS_USE_CUDA
    GTEST_SKIP() << "CUDA not available, skipping Flash Attention test";
#endif

    Tensor q({(size_t)BATCH_SIZE, (size_t)NUM_HEADS, (size_t)SEQ_LEN_Q, (size_t)HEAD_DIM});
    Tensor k({(size_t)BATCH_SIZE, (size_t)NUM_HEADS, (size_t)SEQ_LEN_K, (size_t)HEAD_DIM});
    Tensor v({(size_t)BATCH_SIZE, (size_t)NUM_HEADS, (size_t)SEQ_LEN_K, (size_t)HEAD_DIM});

    q.fill(0.1F);
    k.fill(0.1F);
    v.fill(1.0F);

    q.syncToDevice();
    k.syncToDevice();
    v.syncToDevice();

    FlashAttentionConfig config;
    config.is_causal = false;
    config.softmax_scale = SOFTMAX_SCALE_64;

    Tensor out = flash_attention_forward(q, k, v, config);
    out.syncToHost();

    // With all V=1.0 and constant Q,K, the output should be constant V (1.0)
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out.item(i), 1.0F, TOLERANCE);
    }
}

TEST(FlashAttentionTest, BackwardParity)
{
#ifndef SERIUS_USE_CUDA
    GTEST_SKIP() << "CUDA not available";
#endif

    const int s_small = 64;
    const int d_small = 32;
    Tensor q_data({(size_t)BATCH_SIZE, (size_t)NUM_HEADS, (size_t)s_small, (size_t)d_small});
    Tensor k_data({(size_t)BATCH_SIZE, (size_t)NUM_HEADS, (size_t)s_small, (size_t)d_small});
    Tensor v_data({(size_t)BATCH_SIZE, (size_t)NUM_HEADS, (size_t)s_small, (size_t)d_small});

    q_data.fill(0.1F);
    k_data.fill(0.1F);
    v_data.fill(0.5F);

    q_data.syncToDevice();
    k_data.syncToDevice();
    v_data.syncToDevice();

    auto *q_node = new core::autograd::Node(std::move(q_data));
    auto *k_node = new core::autograd::Node(std::move(k_data));
    auto *v_node = new core::autograd::Node(std::move(v_data));

    q_node->requires_grad = true;
    k_node->requires_grad = true;
    v_node->requires_grad = true;

    FlashAttentionConfig config;
    config.softmax_scale = 1.0F;
    config.is_causal = false;

    auto *out_node = flash_attention(q_node, k_node, v_node, config);

    // Loss = sum(out)
    out_node->grad = Tensor(out_node->data.shape()).fill(1.0F);
    out_node->grad.syncToDevice();

    out_node->backward();

    q_node->grad.syncToHost();
    k_node->grad.syncToHost();
    v_node->grad.syncToHost();

    // Check that gradients are not zero or NaN
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FALSE(std::isnan(q_node->grad.item(i)));
        EXPECT_FALSE(std::isnan(k_node->grad.item(i)));
        EXPECT_FALSE(std::isnan(v_node->grad.item(i)));
    }

    delete out_node;
    delete q_node;
    delete k_node;
    delete v_node;
}

}  // namespace test
}  // namespace serius
