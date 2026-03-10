#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "int4.h"

namespace serius {
namespace quant {

/// GPTQ — Gradient Post-Training Quantization (Frantar et al., 2022)
///
/// GPTQ uses second-order (Hessian) information to minimize quantization error
/// column-by-column. For each column:
///   1. Quantize the column to INT4
///   2. Compute the quantization error delta
///   3. Distribute delta to remaining columns using the inverse Hessian
///      to compensate for the error optimally
///
/// This is based on the Optimal Brain Quantizer (OBQ) framework, made
/// practical via lazy batch updates and Cholesky decomposition of H^{-1}.

/// Configuration for GPTQ quantization
struct GPTQConfig
{
    size_t block_size = 32;       // INT4 quantization block size
    size_t group_size = 128;      // Column group size for Hessian updates
    float damp_percent = 0.01f;   // Diagonal dampening for numerical stability
    bool act_order = false;       // Quantize columns in order of decreasing Hessian diagonal
};

/// Result of GPTQ quantization for a single weight matrix
struct GPTQQuantizedWeight
{
    QuantizedTensor quantized;         // INT4 quantized weights (flattened)
    std::vector<float> group_scales;   // Per-group scale factors
    std::vector<uint8_t> group_zeros;  // Per-group zero points
    size_t rows;
    size_t cols;
    size_t block_size;
};

/// Compute the Hessian approximation from calibration data.
/// H = X^T * X / num_samples, where X is the input activation matrix.
///
/// @param activations  Calibration input activations [num_tokens, K]
/// @param num_tokens   Number of calibration tokens
/// @param K            Input dimension (columns of weight matrix)
/// @return Hessian matrix of shape [K, K] in row-major order
std::vector<float> compute_hessian(const float *activations, size_t num_tokens, size_t K);

/// Quantize a weight matrix using the GPTQ algorithm.
///
/// Algorithm (simplified):
///   For each column j (or column group):
///     1. q_j = quantize(W[:, j])
///     2. error_j = W[:, j] - dequantize(q_j)
///     3. For all remaining columns k > j:
///        W[:, k] += error_j * H_inv[j, k] / H_inv[j, j]
///
/// @param weights     Weight matrix [rows, cols] in row-major order
/// @param rows        Number of output features (rows of W)
/// @param cols        Number of input features (columns of W)
/// @param hessian     Hessian matrix [cols, cols] from compute_hessian()
/// @param config      GPTQ configuration
/// @return GPTQ quantized weight
GPTQQuantizedWeight gptq_quantize(const float *weights, size_t rows, size_t cols,
                                  const std::vector<float> &hessian,
                                  const GPTQConfig &config = GPTQConfig{});

/// Dequantize GPTQ weights back to float (for verification/debugging).
///
/// @param gptq_weight  GPTQ quantized weight
/// @param out          Output buffer [rows, cols]
void gptq_dequantize(const GPTQQuantizedWeight &gptq_weight, float *out);

/// Matrix multiplication with GPTQ-quantized weights.
/// Y = X * W^T where W is stored in GPTQ INT4 format.
///
/// @param gptq_weight  GPTQ quantized weight [N, K]
/// @param X            Input matrix [M, K]
/// @param Y            Output matrix [M, N]
/// @param M            Number of rows in X
void gptq_matmul(const GPTQQuantizedWeight &gptq_weight, const float *X,
                 float *Y, int M);

}  // namespace quant
}  // namespace serius
