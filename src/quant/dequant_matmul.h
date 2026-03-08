#pragma once

#include <cstddef>
#include <cstdint>

namespace serius {
namespace quant {

/// Matrix multiplication with on-the-fly INT4 dequantization (CPU)
///
/// Computes Y = X * W^T where W is stored in INT4 format.
/// De-quantization happens in registers — the quantized weights are
/// never fully expanded to FP32 in memory.
///
/// @param W_q4       Packed INT4 weights, shape [N, K/2] (2 values per byte, row-major)
/// @param scales     Per-block scale factors, shape [N, num_blocks_per_row]
/// @param zeros      Per-block zero points, shape [N, num_blocks_per_row]
/// @param X          Input matrix, shape [M, K] (float32)
/// @param Y          Output matrix, shape [M, N] (float32)
/// @param M          Number of rows in X and Y
/// @param N          Number of rows in W (columns of output)
/// @param K          Number of columns in X / elements per W row
/// @param block_size Quantization block size (e.g. 32)
void dequant_matmul(const uint8_t *W_q4, const float *scales, const uint8_t *zeros, const float *X, float *Y, int M,
                    int N, int K, int block_size);

#ifdef SERIUS_USE_CUDA
/// CUDA version of dequantized matrix multiplication
void dequant_matmul_cuda(const uint8_t *W_q4, const float *scales, const uint8_t *zeros, const float *X, float *Y,
                         int M, int N, int K, int block_size);
#endif

}  // namespace quant
}  // namespace serius
