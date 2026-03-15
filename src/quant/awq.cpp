#include "awq.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace serius {
namespace quant {

ActivationStats collect_activation_stats(const float *activations, size_t num_tokens, size_t num_channels)
{
    ActivationStats stats;
    stats.num_channels = num_channels;
    stats.channel_magnitudes.resize(num_channels, 0.0f);

    if (num_tokens == 0) return stats;

    // Accumulate |activation| per channel across all tokens
    for (size_t t = 0; t < num_tokens; ++t) {
        for (size_t c = 0; c < num_channels; ++c) {
            stats.channel_magnitudes[c] += std::fabs(activations[t * num_channels + c]);
        }
    }

    // Average over tokens
    float inv_tokens = 1.0f / static_cast<float>(num_tokens);
    for (size_t c = 0; c < num_channels; ++c) {
        stats.channel_magnitudes[c] *= inv_tokens;
    }

    return stats;
}

std::vector<float> compute_awq_scales(const ActivationStats &stats, float alpha)
{
    size_t n = stats.num_channels;
    std::vector<float> scales(n, 1.0f);

    if (n == 0) return scales;

    // Find the maximum magnitude to normalize
    float max_mag = *std::max_element(stats.channel_magnitudes.begin(), stats.channel_magnitudes.end());

    if (max_mag <= 0.0f) return scales;

    // AWQ scale formula: s_j = (mag_j / max_mag)^alpha
    // Channels with larger activations get s_j closer to 1 (less scaling needed)
    // Channels with smaller activations get s_j closer to 0 (more compression)
    //
    // But we want to PROTECT salient channels, so we scale them UP:
    // s_j = clamp((mag_j / max_mag)^alpha, min_scale, max_scale)
    //
    // Then W_scaled[:, j] = W[:, j] * s_j (enlarge salient weights)
    // At inference: result[:, j] /= s_j (compensate)

    float inv_max = 1.0f / max_mag;
    for (size_t c = 0; c < n; ++c) {
        float normalized = stats.channel_magnitudes[c] * inv_max;
        // s_j > 1 for salient channels (protects them during quantization)
        // s_j = 1 for the most salient channel
        // We use: s_j = normalized^(-alpha) which gives larger scale for larger activations
        // Actually, the AWQ paper uses: s_j = (activation_magnitude)^alpha as a search parameter
        // Here we implement: s_j = max(normalized, epsilon)^alpha
        // For salient channels: normalized is high, s_j is closer to 1
        // For non-salient: normalized is low, s_j is smaller
        // We then scale weights BY s_j (salient get scaled more = protected)
        float eps = 1e-6f;
        float s = std::pow(std::max(normalized, eps), alpha);
        // Clamp to avoid extreme scales
        scales[c] = std::max(0.01f, std::min(s, 100.0f));
    }

    return scales;
}

AWQQuantizedWeight awq_quantize(const float *weights, size_t rows, size_t cols, const std::vector<float> &scales,
                                size_t block_size)
{
    if (scales.size() != cols) {
        throw std::runtime_error("AWQ: scales size must match number of columns (channels)");
    }

    AWQQuantizedWeight result;
    result.channel_scales = scales;
    result.rows = rows;
    result.cols = cols;

    // Step 1: Apply channel scaling to weights
    // W_scaled[i, j] = W[i, j] * s_j
    std::vector<float> scaled_weights(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            scaled_weights[i * cols + j] = weights[i * cols + j] * scales[j];
        }
    }

    // Step 2: Quantize the scaled weights to INT4 (row by row, block-wise)
    // We quantize the entire flattened scaled weight matrix
    result.quantized = quantize_tensor(scaled_weights.data(), rows * cols, block_size);

    return result;
}

void awq_dequantize(const AWQQuantizedWeight &awq_weight, float *out)
{
    // size_t total = awq_weight.rows * awq_weight.cols;

    // Step 1: Dequantize INT4 to get scaled weights
    dequantize_tensor(awq_weight.quantized, out);

    // Step 2: Undo channel scaling: W_approx[i, j] = W_scaled_dequant[i, j] / s_j
    for (size_t i = 0; i < awq_weight.rows; ++i) {
        for (size_t j = 0; j < awq_weight.cols; ++j) {
            out[i * awq_weight.cols + j] /= awq_weight.channel_scales[j];
        }
    }
}

void awq_matmul(const AWQQuantizedWeight &awq_weight, const float *X, float *Y, int M)
{
    int N = static_cast<int>(awq_weight.rows);
    int K = static_cast<int>(awq_weight.cols);

    // First dequantize weights (INT4 -> float, still scaled)
    std::vector<float> W_scaled(static_cast<size_t>(N) * static_cast<size_t>(K));
    dequantize_tensor(awq_weight.quantized, W_scaled.data());

    // Compute Y = X * W^T with fused scale compensation
    // Y[i, j] = sum_k X[i, k] * (W_scaled[j, k] / s_k)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float w_dequant = W_scaled[j * K + k] / awq_weight.channel_scales[static_cast<size_t>(k)];
                sum += X[i * K + k] * w_dequant;
            }
            Y[i * N + j] = sum;
        }
    }
}

}  // namespace quant
}  // namespace serius
