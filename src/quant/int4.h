#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace serius {
namespace quant {

/// Result of quantizing a block of floats to INT4
struct QuantizedBlock
{
    float scale;
    uint8_t zero_point;
    std::vector<uint8_t> packed;  // Each byte holds 2 x 4-bit values
};

/// Result of quantizing an entire tensor to INT4
struct QuantizedTensor
{
    size_t num_elements;
    size_t block_size;
    std::vector<float> scales;         // One per block
    std::vector<uint8_t> zero_points;  // One per block
    std::vector<uint8_t> packed;       // Packed 4-bit values (2 per byte)
};

// ---- Pack / Unpack ----

/// Pack two 4-bit values (0-15) into a single byte
inline uint8_t pack_two(uint8_t a, uint8_t b)
{
    return (a & 0x0F) | ((b & 0x0F) << 4);
}

/// Unpack a byte into two 4-bit values
inline void unpack_two(uint8_t packed, uint8_t &a, uint8_t &b)
{
    a = packed & 0x0F;
    b = (packed >> 4) & 0x0F;
}

// ---- Block-wise Quantization ----

/// Quantize a block of floats to INT4 (asymmetric, [0,15])
/// @param data   Pointer to float data
/// @param count  Number of elements in this block (may be < block_size for last block)
QuantizedBlock quantize_block(const float *data, size_t count);

/// Dequantize a block back to float
/// @param block  The quantized block
/// @param out    Output float buffer (must have space for block.packed.size()*2 elements)
/// @param count  Number of elements to dequantize
void dequantize_block(const QuantizedBlock &block, float *out, size_t count);

// ---- Tensor-level Quantization ----

/// Quantize a flat float array to INT4 with the given block size
/// @param data       Pointer to float data
/// @param n          Total number of elements
/// @param block_size Number of elements per quantization block (default: 32)
QuantizedTensor quantize_tensor(const float *data, size_t n, size_t block_size = 32);

/// Dequantize an entire QuantizedTensor back to float
/// @param qt   The quantized tensor
/// @param out  Output buffer (must have space for qt.num_elements floats)
void dequantize_tensor(const QuantizedTensor &qt, float *out);

}  // namespace quant
}  // namespace serius
