#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "autograd/node.h"
#include "autograd/ops.h"
#include "optim/amp.h"
#include "tensor/tensor.h"

namespace serius {
namespace test {

using serius::core::Tensor;
using serius::core::autograd::Node;
using serius::optim::AutomaticMixedPrecision;

TEST(AMPTest, LossScaling)
{
    AutomaticMixedPrecision amp({1024.0F, 2000});

    Tensor loss_data({1});
    loss_data.fill(0.5F);
    Node *loss = new Node(std::move(loss_data));

    Node *scaled_loss = amp.scale(loss);

    EXPECT_NEAR(scaled_loss->data.item(0), 0.5F * 1024.0F, 1e-5F);

    delete scaled_loss;
    delete loss;
}

TEST(AMPTest, UnscaleAndOverflow)
{
    // Start with a small scale for easy overflow testing
    AutomaticMixedPrecision amp({1024.0F, 2000});

    Tensor param_data({2, 2});
    param_data.fill(1.0F);
    Node *parameter = new Node(std::move(param_data));
    parameter->requires_grad = true;

    // Test normal unscaling
    parameter->grad.fill(1024.0F);  // grad was scaled by 1024

    std::vector<Node *> params = {parameter};
    bool success = amp.unscale_and_update(params);

    EXPECT_TRUE(success);
    EXPECT_NEAR(parameter->grad.item(0), 1.0F, 1e-5F);

    // Test overflow
    parameter->grad.fill(INFINITY);
    success = amp.unscale_and_update(params);

    EXPECT_FALSE(success);
    // Scale should have backed off
    EXPECT_LT(amp.current_scale(), 1024.0F);
    // Gradients should have been zeroed
    EXPECT_EQ(parameter->grad.item(0), 0.0F);

    delete parameter;
}

TEST(AMPTest, ScaleGrowth)
{
    // Small interval for testing
    AutomaticMixedPrecision amp({100.0F, 2});

    Tensor param_data({1});
    Node *parameter = new Node(std::move(param_data));
    parameter->requires_grad = true;
    std::vector<Node *> params = {parameter};

    // Step 1: No growth yet
    amp.unscale_and_update(params);
    EXPECT_EQ(amp.current_scale(), 100.0F);

    // Step 2: Growth should trigger
    amp.unscale_and_update(params);
    EXPECT_GT(amp.current_scale(), 100.0F);

    delete parameter;
}

}  // namespace test
}  // namespace serius
