#include <gtest/gtest.h>

#include <vector>

#include "hal/device.h"
#include "optim/amp.h"
#include "tensor/tensor.h"

namespace serius {
namespace test {

namespace core = serius::core;
using core::Tensor;
using core::optim::AutomaticMixedPrecision;

TEST(AMPTest, ForwardBackwardScaling)
{
    Tensor input = Tensor::ones({4, 4});  // FP32 input
    Tensor output({1}, hal::DType::Float32);
    Tensor loss({1}, hal::DType::Float32);
    Tensor grad_input({4, 4}, hal::DType::Float32);

    // Dummy forward: out = in * 2
    auto forward_fn = [](const Tensor &in) -> Tensor { return in * 2.0f; };

    // Dummy backward: grad_in = grad_out * 2
    auto backward_fn = [](const Tensor &out_fp16, const Tensor &grad_out, Tensor &grad_in) {
        (void)out_fp16;  // Not used in this dummy
        grad_in = grad_out * 2.0f;
    };

    // Dummy loss: loss = sum(out)
    auto loss_fn = [](const Tensor &out) -> Tensor {
        // Just return a static value 32.0 in FP16 for the test to avoid precision/sum issues
        Tensor l_fp32({1}, hal::DType::Float32);
        l_fp32.data()[0] = 32.0f;
        return l_fp32.cast(hal::DType::Float16);
    };

    // Dummy grad: grad_out = 1
    auto grad_fn = [](const Tensor &loss_val) -> Tensor {
        (void)loss_val;
        return Tensor::ones({4, 4});
    };

    AutomaticMixedPrecision::mixed_precision_forward_backward(forward_fn, backward_fn, loss_fn, grad_fn, input, output,
                                                              loss, grad_input);

    EXPECT_EQ(output.dtype(), hal::DType::Float32);
    EXPECT_FLOAT_EQ(output.data()[0], 2.0f);

    // Initial loss without scale is 16 elements * 2.0 = 32.0
    // Scaled loss should be 32.0 * 65536.0
    Tensor loss_fp32 = loss.cast(hal::DType::Float32);
    EXPECT_FLOAT_EQ(loss_fp32.data()[0], 32.0f * AutomaticMixedPrecision::DEFAULT_LOSS_SCALE);

    // grad_in = grad_out (which is 1) * 2 = 2.0
    Tensor grad_in_fp32 = grad_input.cast(hal::DType::Float32);
    EXPECT_FLOAT_EQ(grad_in_fp32.data()[0], 2.0f);
}

TEST(AMPTest, ConvertModelToFP16)
{
    std::vector<Tensor> model_params;
    model_params.push_back(Tensor::ones({10, 10}));  // FP32

    Tensor t_f16 = Tensor::ones({5, 5});
    model_params.push_back(t_f16.cast(hal::DType::Float16));  // Already FP16

    Tensor t_bf16 = Tensor::ones({2, 2});
    model_params.push_back(t_bf16.cast(hal::DType::BFloat16));  // BF16 (should remain BF16)

    AutomaticMixedPrecision::convert_model_to_fp16(model_params);

    EXPECT_EQ(model_params[0].dtype(), hal::DType::Float16);
    EXPECT_EQ(model_params[1].dtype(), hal::DType::Float16);
    EXPECT_EQ(model_params[2].dtype(), hal::DType::BFloat16);  // Not converted
}

}  // namespace test
}  // namespace serius
