#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "int4.h"

namespace serius {
namespace quant {

/// AWQ — Activation-Aware Weight Quantization (Lin et al., 2023)
///
/// Key insight: not all weights are equally important. Weights corresponding
/// to large activations ("salient channels") contribute more to output quality.
/// AWQ scales salient channels UP before quantization, then scales DOWN at
/// inference time, reducing quantization error where it matters most.
///
/// Advantages over GPTQ:
/// - Faster calibration (no Hessian computation)
/// - Better generalization (less overfitting to calibration set)
/// - Hardware-friendly (simple per-channel scaling)

/// Per-channel activation statistics collected from a calibration set
struct ActivationStats
{
    std::vector<float> channel_magnitudes;  // Average |activation| per channel
    size_t num_channels;
};

/// Result of AWQ quantization
struct AWQQuantizedWeight
{
    QuantizedTensor quantized;           // INT4 quantized weights
    std::vector<float> channel_scales;   // Per-channel AWQ scale factors (s_j)
    size_t rows;                         // Weight matrix rows (output features)
    size_t cols;                         // Weight matrix cols (input features / channels)
};

/// Collect activation statistics from a calibration batch.
/// Processes activations of shape [batch_tokens, channels] and computes
/// the mean absolute value per channel.
///
/// @param activations  Flat array of shape [num_tokens, num_channels]
/// @param num_tokens   Number of tokens in the calibration set
/// @param num_channels Number of input channels (= K dimension of weight matrix)
/// @return Per-channel activation magnitude statistics
ActivationStats collect_activation_stats(const float *activations, size_t num_tokens,
                                         size_t num_channels);

/// Compute AWQ channel scales from activation statistics.
/// Channels with higher activation magnitude get a larger scale factor,
/// which protects them during quantization.
///
/// @param stats            Activation statistics from calibration
/// @param alpha            Scaling strength (0 = no scaling, 1 = full scaling). Default: 0.5
/// @return Per-channel scale factors s_j, shape [num_channels]
std::vector<float> compute_awq_scales(const ActivationStats &stats, float alpha = 0.5f);

/// Quantize a weight matrix using AWQ.
///
/// Algorithm:
/// 1. Scale salient channels: W_scaled[:, j] = W[:, j] * s_j
/// 2. Quantize W_scaled to INT4 (block-wise)
/// 3. Store s_j for inference-time compensation
///
/// @param weights     Weight matrix of shape [rows, cols] in row-major order
/// @param rows        Number of output features
/// @param cols        Number of input features (channels)
/// @param scales      Per-channel AWQ scale factors from compute_awq_scales()
/// @param block_size  INT4 quantization block size (default: 32)
/// @return AWQ quantized weight with embedded scale factors
AWQQuantizedWeight awq_quantize(const float *weights, size_t rows, size_t cols,
                                const std::vector<float> &scales,
                                size_t block_size = 32);

/// Dequantize AWQ weights back to float (for verification/debugging).
/// Applies inverse channel scaling after INT4 dequantization.
///
/// @param awq_weight  AWQ quantized weight
/// @param out         Output buffer of shape [rows, cols]
void awq_dequantize(const AWQQuantizedWeight &awq_weight, float *out);

/// Perform matrix multiplication with AWQ-quantized weights.
/// Y[i,j] = sum_k X[i,k] * (dequant(W_q[j,k]) / s_k)
///
/// The channel scale compensation is fused into the dequantization step
/// to avoid materializing the full FP32 weight matrix.
///
/// @param awq_weight  AWQ quantized weight [N, K]
/// @param X           Input matrix [M, K]
/// @param Y           Output matrix [M, N]
/// @param M           Rows of X
void awq_matmul(const AWQQuantizedWeight &awq_weight, const float *X,
                float *Y, int M);

}  // namespace quant
}  // namespace serius
