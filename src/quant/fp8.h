#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace serius {
namespace quant {

/// FP8 E4M3 format: 1 sign, 4 exponent, 3 mantissa bits
/// Range: [-448, +448], precision suited for weights/activations
/// Used in forward pass on Ada/Hopper architectures (L40, H100)

/// FP8 E5M2 format: 1 sign, 5 exponent, 2 mantissa bits
/// Range: [-57344, +57344], wider range suited for gradients
/// Used in backward pass

// ---- Single-value Conversions ----

/// Convert FP32 to FP8 E4M3 (1-4-3 format)
/// Clamps to [-448, 448], NaN maps to NaN encoding
uint8_t float_to_fp8_e4m3(float f);

/// Convert FP8 E4M3 back to FP32
float fp8_e4m3_to_float(uint8_t v);

/// Convert FP32 to FP8 E5M2 (1-5-2 format)
/// Clamps to [-57344, 57344], NaN maps to NaN encoding
uint8_t float_to_fp8_e5m2(float f);

/// Convert FP8 E5M2 back to FP32
float fp8_e5m2_to_float(uint8_t v);

// ---- Bulk Conversions ----

void convert_float_to_fp8_e4m3(const float *src, uint8_t *dst, size_t n);
void convert_fp8_e4m3_to_float(const uint8_t *src, float *dst, size_t n);

void convert_float_to_fp8_e5m2(const float *src, uint8_t *dst, size_t n);
void convert_fp8_e5m2_to_float(const uint8_t *src, float *dst, size_t n);

// ---- Block-wise Scaling ----

/// Result of FP8 block-wise quantization
struct FP8Block
{
    float scale;                 // Per-block scale factor
    std::vector<uint8_t> data;   // FP8-encoded values (post-scaling)
    bool is_e4m3;                // true = E4M3, false = E5M2
};

/// Quantize a block of floats to FP8 with per-block scaling
/// The block is divided into sub-blocks, each getting its own scale factor
/// to minimize outlier-induced quantization error.
///
/// @param data       Pointer to float data
/// @param count      Number of elements
/// @param use_e4m3   true for E4M3 (weights/activations), false for E5M2 (gradients)
FP8Block fp8_quantize_block(const float *data, size_t count, bool use_e4m3 = true);

/// Dequantize a FP8 block back to float
/// @param block  The FP8 quantized block
/// @param out    Output float buffer (must have space for block.data.size() elements)
void fp8_dequantize_block(const FP8Block &block, float *out);

/// Full tensor FP8 quantization with block-wise scaling
/// @param data        Pointer to float data
/// @param n           Total number of elements
/// @param block_size  Number of elements per scaling block (default: 128)
/// @param use_e4m3    true for E4M3, false for E5M2
/// @param scales_out  Output: per-block scale factors
/// @param fp8_out     Output: FP8-encoded values
void fp8_quantize_tensor(const float *data, size_t n, size_t block_size,
                         bool use_e4m3, std::vector<float> &scales_out,
                         std::vector<uint8_t> &fp8_out);

/// Dequantize a full FP8 tensor back to float
void fp8_dequantize_tensor(const uint8_t *fp8_data, const float *scales,
                           size_t n, size_t block_size, bool use_e4m3,
                           float *out);

}  // namespace quant
}  // namespace serius
