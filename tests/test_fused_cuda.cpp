#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gtest/gtest.h>

#include <iostream>

#include "autograd/node.h"
#include "autograd/ops.h"
#include "hal/device.h"
#include "hal/fused_kernels.h"
#include "tensor/tensor.h"

using namespace serius;
using core::Tensor;
namespace autograd = core::autograd;

class FusedCUDATest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
#ifndef SERIUS_USE_CUDA
        GTEST_SKIP() << "CUDA not enabled";
#endif
    }
};

TEST_F(FusedCUDATest, BiasGELUFP32)
{
    size_t n = 1024;
    size_t bias_size = 128;

    Tensor x_t({n}, hal::DType::Float32);
    Tensor bias_t({bias_size}, hal::DType::Float32);

    // Initialize with test values
    for (size_t i = 0; i < n; ++i) {
        x_t.set_item(i, static_cast<float>(i % 10));
    }
    for (size_t i = 0; i < bias_size; ++i) {
        bias_t.set_item(i, static_cast<float>(i + 1));
    }

    x_t.set_device(hal::DeviceType::CUDA);
    bias_t.set_device(hal::DeviceType::CUDA);

    // Reference implementation
    Tensor ref_t({n}, hal::DType::Float32);
    for (size_t i = 0; i < n; ++i) {
        float val = x_t.item(i) + bias_t.item(i % bias_size);
        float gelu = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        ref_t.set_item(i, gelu);
    }

    // Apply fused kernel
    hal::fused_bias_gelu(x_t.data_gpu(), bias_t.data_gpu(), n, bias_size, hal::DType::Float32, nullptr);

    x_t.syncToHost();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_t.item(i), ref_t.item(i), 1e-5f);
    }
}

TEST_F(FusedCUDATest, BiasGELUFP16)
{
    size_t n = 512;
    size_t bias_size = 64;

    Tensor x_t({n}, hal::DType::Float16);
    Tensor bias_t({bias_size}, hal::DType::Float16);

    for (size_t i = 0; i < n; ++i) {
        x_t.set_item(i, __float2half(static_cast<float>(i % 10)));
    }
    for (size_t i = 0; i < bias_size; ++i) {
        bias_t.set_item(i, __float2half(static_cast<float>(i + 1)));
    }

    x_t.set_device(hal::DeviceType::CUDA);
    bias_t.set_device(hal::DeviceType::CUDA);

    // Reference implementation
    Tensor ref_t({n}, hal::DType::Float32);
    for (size_t i = 0; i < n; ++i) {
        float val = __half2float(x_t.item(i)) + __half2float(bias_t.item(i % bias_size));
        float gelu = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        ref_t.set_item(i, gelu);
    }

    // Apply fused kernel
    hal::fused_bias_gelu(x_t.data_gpu(), bias_t.data_gpu(), n, bias_size, hal::DType::Float16, nullptr);

    x_t.syncToHost();
    for (size_t i = 0; i < n; ++i) {
        float actual = __half2float(x_t.item(i));
        EXPECT_NEAR(actual, ref_t.item(i), 1e-3f);
    }
}

TEST_F(FusedCUDATest, BiasGELUBF16)
{
    size_t n = 512;
    size_t bias_size = 64;

    Tensor x_t({n}, hal::DType::BFloat16);
    Tensor bias_t({bias_size}, hal::DType::BFloat16);

    for (size_t i = 0; i < n; ++i) {
        x_t.set_item(i, __float2bfloat16(static_cast<float>(i % 10)));
    }
    for (size_t i = 0; i < bias_size; ++i) {
        bias_t.set_item(i, __float2bfloat16(static_cast<float>(i + 1)));
    }

    x_t.set_device(hal::DeviceType::CUDA);
    bias_t.set_device(hal::DeviceType::CUDA);

    // Reference implementation
    Tensor ref_t({n}, hal::DType::Float32);
    for (size_t i = 0; i < n; ++i) {
        float val = __bfloat162float(x_t.item(i)) + __bfloat162float(bias_t.item(i % bias_size));
        float gelu = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        ref_t.set_item(i, gelu);
    }

    // Apply fused kernel
    hal::fused_bias_gelu(x_t.data_gpu(), bias_t.data_gpu(), n, bias_size, hal::DType::BFloat16, nullptr);

    x_t.syncToHost();
    for (size_t i = 0; i < n; ++i) {
        float actual = __bfloat162float(x_t.item(i));
        EXPECT_NEAR(actual, ref_t.item(i), 1e-3f);
    }
}

TEST_F(FusedCUDATest, LayerNormForward)
{
    size_t batch = 2;
    size_t seq = 4;
    size_t d_model = 16;

    Tensor input_t({batch, seq, d_model}, hal::DType::Float32);
    input_t.ones();
    input_t.set_device(hal::DeviceType::CUDA);

    autograd::Node input_node(input_t, false);

    Tensor gamma_t({d_model});
    gamma_t.ones();
    autograd::Node gamma(gamma_t, false);

    Tensor beta_t({d_model});
    beta_t.zeros();
    autograd::Node beta(beta_t, false);

    autograd::Node *out = autograd::ops::layernorm(&input_node, &gamma, &beta, 1e-5f);

    out->data.syncToHost();
    for (size_t i = 0; i < out->data.size(); ++i) {
        EXPECT_NEAR(out->data.item(i), 0.0f, 1e-5f);
    }
}

TEST_F(FusedCUDATest, MaskedSoftmaxFP32)
{
    size_t bh = 1;
    size_t sq = 2;
    size_t sk = 4;

    // Shape (B, H, Q, K)
    Tensor input_t({bh, 1, sq, sk}, hal::DType::Float32);
    input_t.ones();
    input_t.set_device(hal::DeviceType::CUDA);

    Tensor mask_t({sq, sk});
    mask_t.zeros();
    // Mask out last two elements of each row
    mask_t.set_item(0, 2, 1.0f);
    mask_t.set_item(0, 3, 1.0f);
    mask_t.set_item(1, 2, 1.0f);
    mask_t.set_item(1, 3, 1.0f);
    mask_t.set_device(hal::DeviceType::CUDA);

    autograd::Node input_node(input_t, false);
    autograd::Node *out = autograd::ops::fused_masked_softmax(&input_node, mask_t, 1.0f);

    out->data.syncToHost();
    // Each row has 4 elements, 2 masked. So row sum is 2. Softmax(1, 1, mask, mask) = (0.5, 0.5, 0, 0)
    for (size_t r = 0; r < sq; ++r) {
        EXPECT_NEAR(out->data.item(0, 0, r, 0), 0.5f, 1e-5f);
        EXPECT_NEAR(out->data.item(0, 0, r, 1), 0.5f, 1e-5f);
        EXPECT_NEAR(out->data.item(0, 0, r, 2), 0.0f, 1e-5f);
        EXPECT_NEAR(out->data.item(0, 0, r, 3), 0.0f, 1e-5f);
    }
    std::cout << "Fused Masked Softmax CUDA: SUCCESS" << std::endl;
}

TEST_F(FusedCUDATest, MatMulFP32)
{
    size_t m = 32;
    size_t n = 32;
    size_t k = 32;

    Tensor a({m, k}, hal::DType::Float32);
    a.ones();
    a.set_device(hal::DeviceType::CUDA);

    Tensor b({k, n}, hal::DType::Float32);
    b.ones();
    b.set_device(hal::DeviceType::CUDA);

    autograd::Node node_a(a, false);
    autograd::Node node_b(b, false);

    autograd::Node *out = autograd::ops::matmul(&node_a, &node_b);

    out->data.syncToHost();
    // Result should be all k (32.0)
    for (size_t i = 0; i < out->data.size(); ++i) {
        EXPECT_NEAR(out->data.item(i), (float)k, 1e-4f);
    }
    std::cout << "CUDA MatMul FP32: SUCCESS" << std::endl;
}

TEST_F(FusedCUDATest, MaskedSoftmaxFP16)
{
    size_t bh = 1;
    size_t sq = 2;
    size_t sk = 4;

    Tensor input_t({bh, 1, sq, sk}, hal::DType::Float16);
    input_t.ones();
    input_t.set_device(hal::DeviceType::CUDA);

    Tensor mask_t({sq, sk});
    mask_t.zeros();
    mask_t.set_item(0, 2, 1.0f);
    mask_t.set_item(0, 3, 1.0f);
    mask_t.set_item(1, 2, 1.0f);
    mask_t.set_item(1, 3, 1.0f);
    mask_t.set_device(hal::DeviceType::CUDA);

    autograd::Node input_node(input_t, false);
    autograd::Node *out = autograd::ops::fused_masked_softmax(&input_node, mask_t, 1.0f);

    out->data.syncToHost();
    for (size_t r = 0; r < sq; ++r) {
        EXPECT_NEAR(out->data.item(0, 0, r, 0), 0.5f, 1e-3f);
        EXPECT_NEAR(out->data.item(0, 0, r, 1), 0.5f, 1e-3f);
        EXPECT_NEAR(out->data.item(0, 0, r, 2), 0.0f, 1e-3f);
        EXPECT_NEAR(out->data.item(0, 0, r, 3), 0.0f, 1e-3f);
    }
    std::cout << "Fused Masked Softmax CUDA FP16: SUCCESS" << std::endl;
}

TEST_F(FusedCUDATest, MaskedSoftmaxBF16)
{
    size_t bh = 1;
    size_t sq = 2;
    size_t sk = 4;

    Tensor input_t({bh, 1, sq, sk}, hal::DType::BFloat16);
    input_t.ones();
    input_t.set_device(hal::DeviceType::CUDA);

    Tensor mask_t({sq, sk});
    mask_t.zeros();
    mask_t.set_item(0, 2, 1.0f);
    mask_t.set_item(0, 3, 1.0f);
    mask_t.set_item(1, 2, 1.0f);
    mask_t.set_item(1, 3, 1.0f);
    mask_t.set_device(hal::DeviceType::CUDA);

    autograd::Node input_node(input_t, false);
    autograd::Node *out = autograd::ops::fused_masked_softmax(&input_node, mask_t, 1.0f);

    out->data.syncToHost();
    for (size_t r = 0; r < sq; ++r) {
        EXPECT_NEAR(out->data.item(0, 0, r, 0), 0.5f, 5e-3f);
        EXPECT_NEAR(out->data.item(0, 0, r, 1), 0.5f, 5e-3f);
        EXPECT_NEAR(out->data.item(0, 0, r, 2), 0.0f, 5e-3f);
        EXPECT_NEAR(out->data.item(0, 0, r, 3), 0.0f, 5e-3f);
    }
    std::cout << "Fused Masked Softmax CUDA BF16: SUCCESS" << std::endl;
}

TEST_F(FusedCUDATest, MatMulFP16)
{
    size_t m = 32;
    size_t n = 32;
    size_t k = 32;

    Tensor a({m, k}, hal::DType::Float16);
    a.ones();
    a.set_device(hal::DeviceType::CUDA);

    Tensor b({k, n}, hal::DType::Float16);
    b.ones();
    b.set_device(hal::DeviceType::CUDA);

    autograd::Node node_a(a, false);
    autograd::Node node_b(b, false);

    autograd::Node *out = autograd::ops::matmul(&node_a, &node_b);

    out->data.syncToHost();
    for (size_t i = 0; i < out->data.size(); ++i) {
        EXPECT_NEAR(out->data.item(i), (float)k, 1e-1f);
    }
    std::cout << "CUDA MatMul FP16: SUCCESS" << std::endl;
}

TEST_F(FusedCUDATest, MatMulBF16)
{
    size_t m = 32;
    size_t n = 32;
    size_t k = 32;

    Tensor a({m, k}, hal::DType::BFloat16);
    a.ones();
    a.set_device(hal::DeviceType::CUDA);

    Tensor b({k, n}, hal::DType::BFloat16);
    b.ones();
    b.set_device(hal::DeviceType::CUDA);

    autograd::Node node_a(a, false);
    autograd::Node node_b(b, false);

    autograd::Node *out = autograd::ops::matmul(&node_a, &node_b);

    out->data.syncToHost();
    for (size_t i = 0; i < out->data.size(); ++i) {
        EXPECT_NEAR(out->data.item(i), (float)k, 1e-1f);
    }
    std::cout << "CUDA MatMul BF16: SUCCESS" << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
