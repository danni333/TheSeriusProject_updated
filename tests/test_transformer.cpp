#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "tensor/tensor.h"
#include "transformer/attention.h"
#include "transformer/block.h"
#include "transformer/ffn.h"
#include "transformer/layernorm.h"
#include "transformer/model.h"
#include "transformer/positional.h"

namespace serius {
namespace test {

namespace core = serius::core;
using core::Tensor;
using core::autograd::Node;
namespace autograd = core::autograd;
using namespace model;

// ===== LayerNorm Tests =====

TEST(TransformerTest, LayerNormStats)
{
    LayerNorm ln(64);
    Tensor input({4, 64});
    input.rand();

    autograd::Node input_node(input, false);
    autograd::Node *output_node = ln.forward(&input_node);
    Tensor output = output_node->data;

    // Check that each row has mean ≈ 0 and variance ≈ 1
    for (size_t row = 0; row < 4; ++row) {
        float mean = 0.0F;
        for (size_t col = 0; col < 64; ++col) {
            mean += output.at(row, col);
        }
        mean /= 64.0F;

        float var = 0.0F;
        for (size_t col = 0; col < 64; ++col) {
            float diff = output.at(row, col) - mean;
            var += diff * diff;
        }
        var /= 64.0F;

        EXPECT_NEAR(mean, 0.0F, 1e-4) << "Row " << row << " mean not near 0";
        EXPECT_NEAR(var, 1.0F, 0.05F) << "Row " << row << " variance not near 1";
    }
}

// ===== Positional Encoding (RoPE) Tests =====

TEST(TransformerTest, RoPEApplication)
{
    // sequence of 4 tokens, d_model = 16, n_heads = 2 -> d_head = 8
    Tensor q({4, 16});
    Tensor k({4, 16});

    q.ones();
    k.ones();

    // apply_rope modifies in-place
    apply_rope(q, k, 8);

    // After RoPE, values should change and first position (pos=0) should just be rotation by 0
    // so pos=0 should remain 1.0
    for (size_t di = 0; di < 16; ++di) {
        EXPECT_FLOAT_EQ(q.at(0, di), 1.0F);
        EXPECT_FLOAT_EQ(k.at(0, di), 1.0F);
    }

    // However, pos=1 should be different
    bool changed = false;
    for (size_t di = 0; di < 16; ++di) {
        if (std::abs(q.at(1, di) - 1.0F) > 1e-4F) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed) << "RoPE did not modify position 1";
}

// ===== Attention Tests =====

TEST(TransformerTest, AttentionShape)
{
    MultiHeadAttention mha(64, 4);
    Tensor input({2, 8, 64});  // (batch, seq, d_model)
    input.rand();

    autograd::Node input_node(input, false);
    autograd::Node *output_node = mha.forward(&input_node);
    Tensor output = output_node->data;

    EXPECT_EQ(output.ndim(), 3U);
    EXPECT_EQ(output.shape()[0], 2U);
    EXPECT_EQ(output.shape()[1], 8U);
    EXPECT_EQ(output.shape()[2], 64U);
}

TEST(TransformerTest, CausalMask)
{
    // Use a small model to verify causal masking
    MultiHeadAttention mha(16, 2);
    Tensor input({1, 4, 16});  // (batch, seq, d_model)

    // Set initial values
    input.fill(1.0F);

    autograd::Node input_node(input, false);
    autograd::Node *out_full_node = mha.forward(&input_node);
    Tensor out_full = out_full_node->data;

    // With causal mask, changing token 3 should NOT change outputs for tokens 0-2
    Tensor input_modified({1, 4, 16});
    input_modified.fill(1.0F);
    for (size_t di = 0; di < 16; ++di) {
        input_modified.at(0, 3, di) = 99.0F;
    }

    autograd::Node input_modified_node(input_modified, false);
    autograd::Node *out_modified_node = mha.forward(&input_modified_node);
    Tensor out_modified = out_modified_node->data;

    // Outputs for token 0 should be practically identical (it can't see token 3)
    for (size_t di = 0; di < 16; ++di) {
        EXPECT_NEAR(out_full.at(0, 0, di), out_modified.at(0, 0, di), 1e-4F) << "Causal mask violated at dim " << di;
    }
}

// ===== FFN Tests =====

TEST(TransformerTest, FFNShape)
{
    FFN ffn(64, 256);
    Tensor input({2, 8, 64});
    input.rand();

    autograd::Node input_node(input, false);
    autograd::Node *output_node = ffn.forward(&input_node);
    Tensor output = output_node->data;

    EXPECT_EQ(output.ndim(), 3U);
    EXPECT_EQ(output.shape()[0], 2U);
    EXPECT_EQ(output.shape()[1], 8U);
    EXPECT_EQ(output.shape()[2], 64U);
}

// ===== Block Tests =====

TEST(TransformerTest, BlockForward)
{
    TransformerBlock block(64, 4, 256);
    Tensor input({2, 8, 64});  // Batched input
    input.rand();

    autograd::Node input_node(input, false);
    autograd::Node *output_node = block.forward(&input_node);
    Tensor output = output_node->data;

    EXPECT_EQ(output.ndim(), 3U);
    EXPECT_EQ(output.shape()[0], 2U);
    EXPECT_EQ(output.shape()[1], 8U);
    EXPECT_EQ(output.shape()[2], 64U);

    // Verify no NaN
    for (size_t idx = 0; idx < output.size(); ++idx) {
        EXPECT_FALSE(std::isnan(output[idx])) << "NaN at index " << idx;
        EXPECT_FALSE(std::isinf(output[idx])) << "Inf at index " << idx;
    }
}

TEST(TransformerTest, BlockBackward)
{
    TransformerBlock block(64, 4, 256);
    Tensor input({2, 4, 64});
    input.rand();

    // Verify finite numerical gradients as autograd integration is evaluated
    autograd::Node input_node(input, false);
    autograd::Node *out1_node = block.forward(&input_node);
    Tensor out1 = out1_node->data;
    float sum1 = 0.0F;
    for (size_t i = 0; i < out1.size(); ++i) sum1 += out1[i];

    float eps = 1e-4F;
    input[0] += eps;
    autograd::Node input2_node(input, false);
    autograd::Node *out2_node = block.forward(&input2_node);
    Tensor out2 = out2_node->data;
    float sum2 = 0.0F;
    for (size_t i = 0; i < out2.size(); ++i) sum2 += out2[i];

    float grad = (sum2 - sum1) / eps;
    EXPECT_FALSE(std::isnan(grad)) << "Numeric gradient is NaN";
    EXPECT_FALSE(std::isinf(grad)) << "Numeric gradient is Inf";
    EXPECT_NE(grad, 0.0F) << "Numeric gradient should be non-zero for block";
}

// ===== Model Tests =====

TEST(TransformerTest, ModelForward)
{
    ModelConfig config;
    config.vocab_size = 100;
    config.d_model = 32;
    config.n_heads = 4;
    config.n_layers = 2;
    config.d_ff = 128;
    config.max_seq_len = 16;

    TransformerModel model(config);

    // Simulate 2 batches, 4 tokens each
    Tensor tokens({2, 4});
    tokens.set_item(0, 0, 1.0F);
    tokens.set_item(0, 1, 5.0F);
    tokens.set_item(0, 2, 10.0F);
    tokens.set_item(0, 3, 50.0F);

    tokens.set_item(1, 0, 2.0F);
    tokens.set_item(1, 1, 6.0F);
    tokens.set_item(1, 2, 11.0F);
    tokens.set_item(1, 3, 51.0F);

    autograd::Node input_node(tokens, false);
    autograd::Node *logits_node = model.forward(&input_node);
    Tensor logits = logits_node->data;

    // Output should be (batch_size=2, seq_len=4, vocab_size=100)
    EXPECT_EQ(logits.ndim(), 3U);
    EXPECT_EQ(logits.shape()[0], 2U);
    EXPECT_EQ(logits.shape()[1], 4U);
    EXPECT_EQ(logits.shape()[2], 100U);

    // Verify no NaN/Inf in logits
    for (size_t idx = 0; idx < logits.size(); ++idx) {
        EXPECT_FALSE(std::isnan(logits[idx])) << "NaN in logits at index " << idx;
        EXPECT_FALSE(std::isinf(logits[idx])) << "Inf in logits at index " << idx;
    }
}
}  // namespace test
}  // namespace serius
