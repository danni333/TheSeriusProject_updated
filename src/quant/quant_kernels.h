#pragma once

#include <cstddef>
#include <cstdint>

namespace serius {
namespace quant {
namespace cuda {

// Single-value conversions (CPU/GPU)
void float_to_fp8_e4m3_cuda(const float* src, uint8_t* dst, size_t n);
void fp8_e4m3_to_float_cuda(const uint8_t* src, float* dst, size_t n);

void float_to_fp8_e5m2_cuda(const float* src, uint8_t* dst, size_t n);
void fp8_e5m2_to_float_cuda(const uint8_t* src, float* dst, size_t n);

// Block-wise quantization on GPU
void fp8_quantize_tensor_cuda(const float* data, uint8_t* fp8_out, float* scales_out, 
                               size_t n, size_t block_size, bool use_e4m3);

void fp8_dequantize_tensor_cuda(const uint8_t* fp8_data, const float* scales, float* out,
                                 size_t n, size_t block_size, bool use_e4m3);

/// Fused FP8 dequant + matmul on device pointers (all args must be GPU memory).
/// Mirrors fp8_dequant_matmul() in fp8.h.
void fp8_dequant_matmul_cuda(const uint8_t* fp8_data, const float* scales,
                             const float* X, float* Y, int M, int N, int K,
                             int block_size, bool use_e4m3);

} // namespace cuda
} // namespace quant
} // namespace serius
