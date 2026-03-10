#include "gptq.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace serius {
namespace quant {

std::vector<float> compute_hessian(const float *activations, size_t num_tokens, size_t K)
{
    // H = X^T * X / num_tokens
    // H is [K, K], symmetric positive semi-definite
    std::vector<float> H(K * K, 0.0f);

    for (size_t t = 0; t < num_tokens; ++t) {
        const float *x = activations + t * K;
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = i; j < K; ++j) {
                float val = x[i] * x[j];
                H[i * K + j] += val;
                if (i != j) {
                    H[j * K + i] += val;
                }
            }
        }
    }

    // Normalize by number of tokens
    float inv_n = 1.0f / static_cast<float>(num_tokens);
    for (size_t i = 0; i < K * K; ++i) {
        H[i] *= inv_n;
    }

    return H;
}

/// Compute the inverse of a symmetric positive definite matrix using
/// Cholesky decomposition. Falls back to diagonal dampening if needed.
static std::vector<float> invert_hessian(const std::vector<float> &H, size_t K, float damp_percent)
{
    std::vector<float> H_damp(H);

    // Add dampening to diagonal for numerical stability
    // damp = damp_percent * mean(diag(H))
    float diag_mean = 0.0f;
    for (size_t i = 0; i < K; ++i) {
        diag_mean += H_damp[i * K + i];
    }
    diag_mean /= static_cast<float>(K);
    float damp = damp_percent * diag_mean;

    for (size_t i = 0; i < K; ++i) {
        H_damp[i * K + i] += damp;
    }

    // Cholesky decomposition: H = L * L^T
    std::vector<float> L(K * K, 0.0f);
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i * K + k] * L[j * K + k];
            }
            if (i == j) {
                float val = H_damp[i * K + i] - sum;
                if (val <= 0.0f) {
                    // Not positive definite, add more dampening
                    val = damp;
                }
                L[i * K + j] = std::sqrt(val);
            } else {
                L[i * K + j] = (H_damp[i * K + j] - sum) / L[j * K + j];
            }
        }
    }

    // Invert L (forward substitution on identity)
    std::vector<float> L_inv(K * K, 0.0f);
    for (size_t i = 0; i < K; ++i) {
        L_inv[i * K + i] = 1.0f / L[i * K + i];
        for (size_t j = i + 1; j < K; ++j) {
            float sum = 0.0f;
            for (size_t k = i; k < j; ++k) {
                sum += L[j * K + k] * L_inv[k * K + i];
            }
            L_inv[j * K + i] = -sum / L[j * K + j];
        }
    }

    // H^{-1} = (L^T)^{-1} * L^{-1} = L_inv^T * L_inv
    std::vector<float> H_inv(K * K, 0.0f);
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = i; j < K; ++j) {
            float sum = 0.0f;
            // Only non-zero where k >= max(i, j) = j (since L_inv is lower triangular)
            for (size_t k = j; k < K; ++k) {
                sum += L_inv[k * K + i] * L_inv[k * K + j];
            }
            H_inv[i * K + j] = sum;
            H_inv[j * K + i] = sum;
        }
    }

    return H_inv;
}

/// Quantize a single value to INT4 with given scale and zero_point
static uint8_t quantize_single(float val, float scale, uint8_t zero_point)
{
    if (scale == 0.0f) return zero_point;
    float inv_scale = 1.0f / scale;
    int q = static_cast<int>(std::round(val * inv_scale + static_cast<float>(zero_point)));
    q = std::max(0, std::min(15, q));
    return static_cast<uint8_t>(q);
}

/// Dequantize a single INT4 value
static float dequantize_single(uint8_t q, float scale, uint8_t zero_point)
{
    return (static_cast<float>(q) - static_cast<float>(zero_point)) * scale;
}

GPTQQuantizedWeight gptq_quantize(const float *weights, size_t rows, size_t cols,
                                  const std::vector<float> &hessian,
                                  const GPTQConfig &config)
{
    if (hessian.size() != cols * cols) {
        throw std::runtime_error("GPTQ: Hessian size must be cols*cols");
    }

    GPTQQuantizedWeight result;
    result.rows = rows;
    result.cols = cols;
    result.block_size = config.block_size;

    // Compute H^{-1}
    std::vector<float> H_inv = invert_hessian(hessian, cols, config.damp_percent);

    // Working copy of weights (will be modified during quantization)
    std::vector<float> W(weights, weights + rows * cols);

    // Output quantized values (flat, one per weight element)
    std::vector<uint8_t> q_values(rows * cols, 0);

    // Group-wise scales and zeros
    size_t num_groups_per_row = (cols + config.block_size - 1) / config.block_size;
    size_t total_groups = rows * num_groups_per_row;
    result.group_scales.resize(total_groups);
    result.group_zeros.resize(total_groups);

    // Pre-compute group parameters for each row
    for (size_t i = 0; i < rows; ++i) {
        for (size_t g = 0; g < num_groups_per_row; ++g) {
            size_t start = g * config.block_size;
            size_t end = std::min(start + config.block_size, cols);
            size_t count = end - start;

            // Find min/max for this group in the ORIGINAL weights
            float w_min = weights[i * cols + start];
            float w_max = w_min;
            for (size_t j = start + 1; j < end; ++j) {
                float val = weights[i * cols + j];
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }

            float scale;
            uint8_t zero_point;
            if (w_max == w_min) {
                scale = 1.0f;
                zero_point = 0;
            } else {
                scale = (w_max - w_min) / 15.0f;
                int zp = static_cast<int>(std::round(-w_min / scale));
                zp = std::max(0, std::min(15, zp));
                zero_point = static_cast<uint8_t>(zp);
            }

            size_t group_idx = i * num_groups_per_row + g;
            result.group_scales[group_idx] = scale;
            result.group_zeros[group_idx] = zero_point;
        }
    }

    // GPTQ: Process columns left to right, compensating errors
    // For efficiency, process in column groups
    size_t group_cols = config.group_size;

    for (size_t col_start = 0; col_start < cols; col_start += group_cols) {
        size_t col_end = std::min(col_start + group_cols, cols);

        for (size_t j = col_start; j < col_end; ++j) {
            float h_inv_jj = H_inv[j * cols + j];
            if (h_inv_jj <= 0.0f) h_inv_jj = 1e-6f;

            for (size_t i = 0; i < rows; ++i) {
                float w_val = W[i * cols + j];

                // Determine which group this column belongs to
                size_t g = j / config.block_size;
                size_t group_idx = i * num_groups_per_row + g;
                float scale = result.group_scales[group_idx];
                uint8_t zero = result.group_zeros[group_idx];

                // Quantize
                uint8_t q = quantize_single(w_val, scale, zero);
                q_values[i * cols + j] = q;

                // Compute quantization error
                float w_quant = dequantize_single(q, scale, zero);
                float error = w_val - w_quant;

                // Distribute error to remaining columns in the current group
                // W[:, k] += error * H_inv[j, k] / H_inv[j, j]
                float scale_factor = error / h_inv_jj;
                for (size_t k = j + 1; k < col_end; ++k) {
                    W[i * cols + k] += scale_factor * H_inv[j * cols + k];
                }
            }
        }
    }

    // Pack quantized values into INT4 format
    size_t total_packed = (rows * cols + 1) / 2;
    std::vector<uint8_t> packed(total_packed);
    for (size_t idx = 0; idx < rows * cols; idx += 2) {
        uint8_t a = q_values[idx];
        uint8_t b = (idx + 1 < rows * cols) ? q_values[idx + 1] : 0;
        packed[idx / 2] = pack_two(a, b);
    }

    // Build the QuantizedTensor
    result.quantized.num_elements = rows * cols;
    result.quantized.block_size = config.block_size;
    result.quantized.scales = result.group_scales;
    result.quantized.zero_points = result.group_zeros;
    result.quantized.packed = std::move(packed);

    return result;
}

void gptq_dequantize(const GPTQQuantizedWeight &gptq_weight, float *out)
{
    size_t rows = gptq_weight.rows;
    size_t cols = gptq_weight.cols;
    size_t block_size = gptq_weight.block_size;
    size_t num_groups_per_row = (cols + block_size - 1) / block_size;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t flat_idx = i * cols + j;
            size_t packed_idx = flat_idx / 2;
            uint8_t packed = gptq_weight.quantized.packed[packed_idx];
            uint8_t q;
            if (flat_idx % 2 == 0) {
                q = packed & 0x0F;
            } else {
                q = (packed >> 4) & 0x0F;
            }

            size_t g = j / block_size;
            size_t group_idx = i * num_groups_per_row + g;
            float scale = gptq_weight.group_scales[group_idx];
            uint8_t zero = gptq_weight.group_zeros[group_idx];

            out[flat_idx] = dequantize_single(q, scale, zero);
        }
    }
}

void gptq_matmul(const GPTQQuantizedWeight &gptq_weight, const float *X,
                 float *Y, int M)
{
    int N = static_cast<int>(gptq_weight.rows);
    int K = static_cast<int>(gptq_weight.cols);
    size_t block_size = gptq_weight.block_size;
    size_t num_groups_per_row = (static_cast<size_t>(K) + block_size - 1) / block_size;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                size_t flat_idx = static_cast<size_t>(j) * static_cast<size_t>(K) + static_cast<size_t>(k);
                size_t packed_idx = flat_idx / 2;
                uint8_t packed = gptq_weight.quantized.packed[packed_idx];
                uint8_t q;
                if (flat_idx % 2 == 0) {
                    q = packed & 0x0F;
                } else {
                    q = (packed >> 4) & 0x0F;
                }

                size_t g = static_cast<size_t>(k) / block_size;
                size_t group_idx = static_cast<size_t>(j) * num_groups_per_row + g;
                float scale = gptq_weight.group_scales[group_idx];
                uint8_t zero = gptq_weight.group_zeros[group_idx];

                float w = dequantize_single(q, scale, zero);
                sum += X[i * K + k] * w;
            }
            Y[i * N + j] = sum;
        }
    }
}

}  // namespace quant
}  // namespace serius
