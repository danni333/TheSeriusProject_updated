#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "attention/flash_attn.h"
#include "autograd/engine.h"
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

TEST(FlashAttentionTest, ForwardMatchesStableReferenceForCausalAndPartialBlocks)
{
#ifndef SERIUS_USE_CUDA
    GTEST_SKIP() << "CUDA not available, skipping Flash Attention test";
#else
    constexpr int batch_size = 2;
    constexpr int num_heads = 2;
    constexpr int seq_len_q = 5;
    constexpr int seq_len_k = 7;
    constexpr int head_dim = 32;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const size_t q_count = batch_size * num_heads * seq_len_q * head_dim;
    const size_t kv_count = batch_size * num_heads * seq_len_k * head_dim;

    Tensor q({batch_size, num_heads, seq_len_q, head_dim});
    Tensor k({batch_size, num_heads, seq_len_k, head_dim});
    Tensor v({batch_size, num_heads, seq_len_k, head_dim});
    std::vector<float> q_values(q_count);
    std::vector<float> k_values(kv_count);
    std::vector<float> v_values(kv_count);
    for (size_t i = 0; i < q_count; ++i) q_values[i] = std::sin(static_cast<float>(i) * 0.17F);
    for (size_t i = 0; i < kv_count; ++i) {
        k_values[i] = std::cos(static_cast<float>(i) * 0.11F);
        v_values[i] = std::sin(static_cast<float>(i) * 0.07F + 0.3F);
    }
    for (size_t i = 0; i < q_count; ++i) q[i] = q_values[i];
    for (size_t i = 0; i < kv_count; ++i) {
        k[i] = k_values[i];
        v[i] = v_values[i];
    }
    q.syncToDevice();
    k.syncToDevice();
    v.syncToDevice();

    const std::vector<std::pair<int, int>> block_sizes = {{0, 0}, {16, 32}, {64, 16}, {128, 64}};
    for (bool is_causal : {false, true}) {
        for (const auto &[block_m, block_n] : block_sizes) {
            FlashAttentionConfig config;
            config.is_causal = is_causal;
            config.softmax_scale = scale;
            config.block_m = block_m;
            config.block_n = block_n;
            Tensor output = flash_attention_forward(q, k, v, config);
            output.syncToHost();

            for (int batch = 0; batch < batch_size; ++batch) {
                for (int head = 0; head < num_heads; ++head) {
                    for (int q_idx = 0; q_idx < seq_len_q; ++q_idx) {
                        std::vector<float> scores(seq_len_k, -std::numeric_limits<float>::infinity());
                        float max_score = -std::numeric_limits<float>::infinity();
                        for (int k_idx = 0; k_idx < seq_len_k; ++k_idx) {
                            if (is_causal && q_idx < k_idx) continue;
                            float score = 0.0F;
                            for (int dim = 0; dim < head_dim; ++dim) {
                                const size_t q_offset =
                                    ((batch * num_heads + head) * seq_len_q + q_idx) * head_dim + dim;
                                const size_t k_offset =
                                    ((batch * num_heads + head) * seq_len_k + k_idx) * head_dim + dim;
                                score += q_values[q_offset] * k_values[k_offset];
                            }
                            scores[k_idx] = score * scale;
                            max_score = std::max(max_score, scores[k_idx]);
                        }

                        float denominator = 0.0F;
                        for (int k_idx = 0; k_idx < seq_len_k; ++k_idx) {
                            if (std::isfinite(scores[k_idx])) denominator += std::exp(scores[k_idx] - max_score);
                        }
                        for (int dim = 0; dim < head_dim; ++dim) {
                            float expected = 0.0F;
                            for (int k_idx = 0; k_idx < seq_len_k; ++k_idx) {
                                if (!std::isfinite(scores[k_idx])) continue;
                                const size_t v_offset =
                                    ((batch * num_heads + head) * seq_len_k + k_idx) * head_dim + dim;
                                expected += std::exp(scores[k_idx] - max_score) * v_values[v_offset] / denominator;
                            }
                            const size_t out_offset =
                                ((batch * num_heads + head) * seq_len_q + q_idx) * head_dim + dim;
                            EXPECT_NEAR(output[out_offset], expected, 2e-4F);
                        }
                    }
                }
            }
        }
    }
#endif
}

TEST(FlashAttentionTest, RejectsIncompatibleValueShape)
{
    Tensor q({1, 1, 2, 32});
    Tensor k({1, 1, 3, 32});
    Tensor v({1, 1, 4, 32});
    EXPECT_THROW(flash_attention_forward(q, k, v, FlashAttentionConfig{}), std::invalid_argument);
}

TEST(FlashAttentionTest, RejectsUnsupportedHeadDimension)
{
    Tensor q({1, 1, 2, 16});
    Tensor k({1, 1, 2, 16});
    Tensor v({1, 1, 2, 16});
    EXPECT_THROW(flash_attention_forward(q, k, v, FlashAttentionConfig{}), std::invalid_argument);
}

TEST(FlashAttentionTest, RejectsInvalidBlockConfiguration)
{
    Tensor q({1, 1, 2, 64});
    Tensor k({1, 1, 2, 64});
    Tensor v({1, 1, 2, 64});
    FlashAttentionConfig config;
    config.block_m = 128;
    config.block_n = 128;
    EXPECT_THROW(flash_attention_forward(q, k, v, config), std::invalid_argument);
}

TEST(FlashAttentionTest, SupportsConfiguredTilesForEveryHeadDimension)
{
#ifndef SERIUS_USE_CUDA
    GTEST_SKIP() << "CUDA not available, skipping Flash Attention test";
#else
    for (int head_dim : {32, 64, 128}) {
        Tensor q({1, 1, 3, static_cast<size_t>(head_dim)});
        Tensor k({1, 1, 5, static_cast<size_t>(head_dim)});
        Tensor v({1, 1, 5, static_cast<size_t>(head_dim)});
        for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(static_cast<float>(i) * 0.13F);
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = std::cos(static_cast<float>(i) * 0.11F);
            v[i] = std::sin(static_cast<float>(i) * 0.07F);
        }
        q.syncToDevice();
        k.syncToDevice();
        v.syncToDevice();

        FlashAttentionConfig default_config;
        default_config.is_causal = false;
        default_config.softmax_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        Tensor expected = flash_attention_forward(q, k, v, default_config);
        expected.syncToHost();

        FlashAttentionConfig configured = default_config;
        configured.block_m = head_dim == 32 ? 64 : 16;
        configured.block_n = head_dim == 32 ? 32 : 32;
        Tensor actual = flash_attention_forward(q, k, v, configured);
        actual.syncToHost();
        for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 2e-4F);
    }
#endif
}

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
#else
    constexpr int seq_len_q = 3;
    constexpr int seq_len_k = 4;
    constexpr int head_dim = 32;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const size_t q_count = seq_len_q * head_dim;
    const size_t kv_count = seq_len_k * head_dim;

    Tensor q_data({1, 1, seq_len_q, head_dim});
    Tensor k_data({1, 1, seq_len_k, head_dim});
    Tensor v_data({1, 1, seq_len_k, head_dim});
    std::vector<float> q_values(q_count);
    std::vector<float> k_values(kv_count);
    std::vector<float> v_values(kv_count);
    std::vector<float> grad_output(q_count);
    for (size_t i = 0; i < q_count; ++i) {
        q_values[i] = std::sin(static_cast<float>(i) * 0.13F);
        grad_output[i] = std::cos(static_cast<float>(i) * 0.09F);
        q_data[i] = q_values[i];
    }
    for (size_t i = 0; i < kv_count; ++i) {
        k_values[i] = std::cos(static_cast<float>(i) * 0.15F);
        v_values[i] = std::sin(static_cast<float>(i) * 0.08F + 0.2F);
        k_data[i] = k_values[i];
        v_data[i] = v_values[i];
    }
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
    config.softmax_scale = scale;
    config.is_causal = true;
    auto *out_node = flash_attention(q_node, k_node, v_node, config);

    out_node->grad = core::Tensor(out_node->data.shape());
    for (size_t i = 0; i < q_count; ++i) out_node->grad[i] = grad_output[i];
    out_node->grad.syncToDevice();

    if (out_node->backward_fn) {
        out_node->backward_fn();
    }

    q_node->grad.syncToHost();
    k_node->grad.syncToHost();
    v_node->grad.syncToHost();

    std::vector<float> expected_q_grad(q_count, 0.0F);
    std::vector<float> expected_k_grad(kv_count, 0.0F);
    std::vector<float> expected_v_grad(kv_count, 0.0F);
    for (int q_idx = 0; q_idx < seq_len_q; ++q_idx) {
        std::vector<float> scores(seq_len_k, -std::numeric_limits<float>::infinity());
        float max_score = -std::numeric_limits<float>::infinity();
        for (int k_idx = 0; k_idx <= q_idx; ++k_idx) {
            float score = 0.0F;
            for (int dim = 0; dim < head_dim; ++dim) {
                score += q_values[q_idx * head_dim + dim] * k_values[k_idx * head_dim + dim];
            }
            scores[k_idx] = score * scale;
            max_score = std::max(max_score, scores[k_idx]);
        }
        float denominator = 0.0F;
        for (int k_idx = 0; k_idx <= q_idx; ++k_idx) denominator += std::exp(scores[k_idx] - max_score);
        std::vector<float> probabilities(seq_len_k, 0.0F);
        for (int k_idx = 0; k_idx <= q_idx; ++k_idx) {
            probabilities[k_idx] = std::exp(scores[k_idx] - max_score) / denominator;
        }

        std::vector<float> output(head_dim, 0.0F);
        float delta = 0.0F;
        for (int k_idx = 0; k_idx <= q_idx; ++k_idx) {
            for (int dim = 0; dim < head_dim; ++dim) {
                output[dim] += probabilities[k_idx] * v_values[k_idx * head_dim + dim];
            }
        }
        for (int dim = 0; dim < head_dim; ++dim) delta += grad_output[q_idx * head_dim + dim] * output[dim];

        for (int k_idx = 0; k_idx <= q_idx; ++k_idx) {
            float dp = 0.0F;
            for (int dim = 0; dim < head_dim; ++dim) {
                dp += grad_output[q_idx * head_dim + dim] * v_values[k_idx * head_dim + dim];
                expected_v_grad[k_idx * head_dim + dim] +=
                    probabilities[k_idx] * grad_output[q_idx * head_dim + dim];
            }
            const float ds = probabilities[k_idx] * (dp - delta);
            for (int dim = 0; dim < head_dim; ++dim) {
                expected_q_grad[q_idx * head_dim + dim] += scale * ds * k_values[k_idx * head_dim + dim];
                expected_k_grad[k_idx * head_dim + dim] += scale * ds * q_values[q_idx * head_dim + dim];
            }
        }
    }
    for (size_t i = 0; i < q_count; ++i) {
        EXPECT_NEAR(q_node->grad[i], expected_q_grad[i], 2e-4F);
    }
    for (size_t i = 0; i < kv_count; ++i) {
        EXPECT_NEAR(k_node->grad[i], expected_k_grad[i], 2e-4F);
        EXPECT_NEAR(v_node->grad[i], expected_v_grad[i], 2e-4F);
    }

    delete out_node;
    delete q_node;
    delete k_node;
    delete v_node;
#endif
}

}  // namespace test
}  // namespace serius
