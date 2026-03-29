#include <cuda_fp16.h>
#ifdef SERIUS_USE_CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#endif
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
    x_t.mark_dirty(Tensor::DataLocation::GPU);

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
        float val = x_t.item(i) + bias_t.item(i % bias_size);
        float gelu = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        ref_t.set_item(i, gelu);
    }

    // Apply fused kernel
    hal::fused_bias_gelu(x_t.data_gpu(), bias_t.data_gpu(), n, bias_size, hal::DType::Float16, nullptr);
    x_t.mark_dirty(Tensor::DataLocation::GPU);

    x_t.syncToHost();
    for (size_t i = 0; i < n; ++i) {
        float actual = x_t.item(i);
        EXPECT_NEAR(actual, ref_t.item(i), 1e-2f);
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
        float val = x_t.item(i) + bias_t.item(i % bias_size);
        float gelu = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        ref_t.set_item(i, gelu);
    }

    // Apply fused kernel
    hal::fused_bias_gelu(x_t.data_gpu(), bias_t.data_gpu(), n, bias_size, hal::DType::BFloat16, nullptr);
    x_t.mark_dirty(Tensor::DataLocation::GPU);

    x_t.syncToHost();
    for (size_t i = 0; i < n; ++i) {
        float actual = x_t.item(i);
        EXPECT_NEAR(actual, ref_t.item(i), 0.5f);
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

TEST_F(FusedCUDATest, RMSNormFP32)
{
    size_t batch_seq = 2;
    size_t n = 1024;
    float eps = 1e-5f;

    Tensor input({batch_seq, n}, hal::DType::Float32);
    Tensor weight({n}, hal::DType::Float32);
    Tensor out({batch_seq, n}, hal::DType::Float32);

    input.ones();
    weight.ones();
    input.set_device(hal::DeviceType::CUDA);
    weight.set_device(hal::DeviceType::CUDA);
    out.set_device(hal::DeviceType::CUDA);

    hal::rmsnorm(out.data_gpu(), input.data_gpu(), weight.data_gpu(), n, batch_seq, eps, hal::DType::Float32);
    out.mark_dirty(Tensor::DataLocation::GPU);

    out.syncToHost();
    // RMS of [1, 1, ...] is 1. Result should be 1.0
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out.item(i), 1.0f, 1e-5f);
    }
    std::cout << "CUDA RMSNorm FP32: SUCCESS" << std::endl;
}

TEST_F(FusedCUDATest, SwiGLUFP32)
{
    size_t n = 1024;
    Tensor x({n}, hal::DType::Float32);
    Tensor y({n}, hal::DType::Float32);
    Tensor out({n}, hal::DType::Float32);

    x.ones(); // x = 1.0
    y.set_item(0, 2.0f); // y = 2.0
    for(size_t i=1; i<n; ++i) y.set_item(i, 2.0f);
    
    x.set_device(hal::DeviceType::CUDA);
    y.set_device(hal::DeviceType::CUDA);
    out.set_device(hal::DeviceType::CUDA);

    hal::swiglu(out.data_gpu(), x.data_gpu(), y.data_gpu(), n, hal::DType::Float32);
    out.mark_dirty(Tensor::DataLocation::GPU);

    out.syncToHost();
    // silu(1.0) * 2.0 = (1.0 / (1 + exp(-1))) * 2.0 = 0.731058 * 2 = 1.462117
    float expected = (1.0f / (1.0f + expf(-1.0f))) * 2.0f;
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out.item(i), expected, 1e-5f);
    }
    std::cout << "CUDA SwiGLU FP32: SUCCESS" << std::endl;
}

TEST_F(FusedCUDATest, RoPEFP32)
{
    size_t head_dim = 64;
    size_t seq_len = 2;
    size_t n_heads = 2;
    
    Tensor data({seq_len, n_heads, head_dim}, hal::DType::Float32);
    data.ones();
    data.set_device(hal::DeviceType::CUDA);

    hal::rope(data.data_gpu(), head_dim, seq_len, n_heads, 10000.0f, 0, hal::DType::Float32);
    data.mark_dirty(Tensor::DataLocation::GPU);

    data.syncToHost();
    // At pos 0, theta = 0, cos=1, sin=0. x0' = 1*1 - 1*0 = 1, x1' = 1*0 + 1*1 = 1.
    // Check pos 0
    for (size_t i = 0; i < n_heads * head_dim; ++i) {
        EXPECT_NEAR(data.item(0, i / head_dim, i % head_dim), 1.0f, 1e-5f);
    }
    std::cout << "CUDA RoPE FP32: SUCCESS" << std::endl;
}

TEST_F(FusedCUDATest, SoftmaxRowWiseFP32)
{
    size_t rows = 2;
    size_t cols = 4;
    Tensor x({rows, cols}, hal::DType::Float32);
    x.ones(); 
    x.set_device(hal::DeviceType::CUDA);

    hal::softmax(x.data_gpu(), rows, cols, hal::DType::Float32);
    x.mark_dirty(Tensor::DataLocation::GPU);

    x.syncToHost();
    // Softmax of [1, 1, 1, 1] is [0.25, 0.25, 0.25, 0.25]
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x.item(i), 0.25f, 1e-5f);
    }
    std::cout << "CUDA Softmax Row-wise FP32: SUCCESS" << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
