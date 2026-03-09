#include <gtest/gtest.h>

#include "autograd/engine.h"
#include "tensor/tensor.h"
#include "transformer/block.h"


namespace serius {
namespace test {

using core::Tensor;
using core::autograd::AutogradEngine;
using core::autograd::Node;
using model::TransformerBlock;

TEST(TransformerOptimTest, ForwardMatchesOptimized)
{
    size_t d_model = 64;
    size_t n_heads = 4;
    size_t d_ff = 256;
    size_t seq_len = 10;
    size_t batch_size = 2;

    TransformerBlock block(d_model, n_heads, d_ff);

    // Initialize block parameters deterministically for testing
    // LayerNorm
    for (size_t i = 0; i < d_model; ++i) {
        block.norm1.gamma->data[i] = 1.0f + 0.01f * i;
        block.norm1.beta->data[i] = 0.05f * i;
        block.norm2.gamma->data[i] = 1.0f - 0.01f * i;
        block.norm2.beta->data[i] = -0.05f * i;
    }

    // FFN
    block.ffn.w1->data.rand();
    block.ffn.b1->data.rand();
    block.ffn.w2->data.rand();
    block.ffn.b2->data.rand();

    // Attention parameters (already initialized with xavier, let's keep them)
    // We just need the output to match

    Tensor input_data({batch_size, seq_len, d_model});
    input_data.rand();

    Node *input_node1 = new Node(input_data);
    Node *input_node2 = new Node(input_data);

    // Standard forward
    Node *out_standard = block.forward(input_node1);

    // Optimized forward
    Node *out_optimized = block.forward_optimized(input_node2);

    // Verify sizes
    EXPECT_EQ(out_standard->data.shape(), out_optimized->data.shape());

    // Verify values block by block
    // Floating point math with varying sequence of operations can introduce tiny differences
    float max_diff = 0.0f;
    for (size_t i = 0; i < out_standard->data.size(); ++i) {
        float diff = std::abs(out_standard->data[i] - out_optimized->data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    // Tolerance might need to be larger due to GELU fast math approx
    // Wait, the standard GELU also uses fast math approx!
    // Actually standard gelu and fused bias+gelu use the exact same approximation.
    // So the difference should be extremely small, just floating point reordering.
    EXPECT_LT(max_diff, 1e-4f) << "Max difference between standard and optimized block too high: " << max_diff;

    // Cleanup
    // In a real scenario AutogradEngine cleans up, let's just do it manual here to avoid memory leak warnings
    delete input_node1;
    delete input_node2;
    delete out_standard;
    delete out_optimized;
}

}  // namespace test
}  // namespace serius
