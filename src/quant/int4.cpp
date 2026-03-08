#include "int4.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace serius {
namespace quant {

QuantizedBlock quantize_block(const float *data, size_t count)
{
    QuantizedBlock block;

    // Find min/max of the block
    float x_min = data[0];
    float x_max = data[0];
    for (size_t i = 1; i < count; ++i) {
        x_min = std::min(x_min, data[i]);
        x_max = std::max(x_max, data[i]);
    }

    // Handle degenerate case where all values are the same
    if (x_max == x_min) {
        block.scale = 1.0f;
        block.zero_point = 0;
        // All values quantize to the same thing
        size_t packed_count = (count + 1) / 2;
        block.packed.resize(packed_count, 0);
        return block;
    }

    // Asymmetric quantization to [0, 15]
    block.scale = (x_max - x_min) / 15.0f;
    float inv_scale = 1.0f / block.scale;

    // Compute zero_point: q = round((x - x_min) / scale)
    // so zero_point corresponds to x = 0 → zp = round(-x_min / scale)
    // but clamped to [0, 15]
    int zp = static_cast<int>(std::round(-x_min * inv_scale));
    zp = std::max(0, std::min(15, zp));
    block.zero_point = static_cast<uint8_t>(zp);

    // Quantize and pack
    size_t packed_count = (count + 1) / 2;
    block.packed.resize(packed_count);

    for (size_t i = 0; i < count; i += 2) {
        int q0 = static_cast<int>(std::round((data[i] - x_min) * inv_scale));
        q0 = std::max(0, std::min(15, q0));

        int q1 = 0;
        if (i + 1 < count) {
            q1 = static_cast<int>(std::round((data[i + 1] - x_min) * inv_scale));
            q1 = std::max(0, std::min(15, q1));
        }

        block.packed[i / 2] = pack_two(static_cast<uint8_t>(q0), static_cast<uint8_t>(q1));
    }

    return block;
}

void dequantize_block(const QuantizedBlock &block, float *out, size_t count)
{
    // Dequantize: x_approx = (q - zero_point) * scale
    // But since we used (x - x_min) / scale for quantization,
    // the dequantization is: x_approx = q * scale + x_min
    // And x_min = -zero_point * scale
    // So: x_approx = (q - zero_point) * scale
    for (size_t i = 0; i < count; i += 2) {
        uint8_t a, b;
        unpack_two(block.packed[i / 2], a, b);

        out[i] = (static_cast<float>(a) - static_cast<float>(block.zero_point)) * block.scale;
        if (i + 1 < count) {
            out[i + 1] = (static_cast<float>(b) - static_cast<float>(block.zero_point)) * block.scale;
        }
    }
}

QuantizedTensor quantize_tensor(const float *data, size_t n, size_t block_size)
{
    QuantizedTensor qt;
    qt.num_elements = n;
    qt.block_size = block_size;

    size_t num_blocks = (n + block_size - 1) / block_size;
    qt.scales.resize(num_blocks);
    qt.zero_points.resize(num_blocks);

    // Total packed bytes: each block packs block_size elements → block_size/2 bytes
    size_t total_packed = (n + 1) / 2;
    qt.packed.resize(total_packed);

    size_t packed_offset = 0;
    for (size_t b = 0; b < num_blocks; ++b) {
        size_t start = b * block_size;
        size_t count = std::min(block_size, n - start);

        QuantizedBlock block = quantize_block(data + start, count);
        qt.scales[b] = block.scale;
        qt.zero_points[b] = block.zero_point;

        // Copy packed data
        std::copy(block.packed.begin(), block.packed.end(),
                  qt.packed.begin() + static_cast<std::ptrdiff_t>(packed_offset));
        packed_offset += block.packed.size();
    }

    return qt;
}

void dequantize_tensor(const QuantizedTensor &qt, float *out)
{
    size_t num_blocks = (qt.num_elements + qt.block_size - 1) / qt.block_size;
    size_t packed_offset = 0;

    for (size_t b = 0; b < num_blocks; ++b) {
        size_t start = b * qt.block_size;
        size_t count = std::min(qt.block_size, qt.num_elements - start);

        QuantizedBlock block;
        block.scale = qt.scales[b];
        block.zero_point = qt.zero_points[b];

        size_t block_packed = (count + 1) / 2;
        block.packed.assign(qt.packed.begin() + static_cast<std::ptrdiff_t>(packed_offset),
                            qt.packed.begin() + static_cast<std::ptrdiff_t>(packed_offset + block_packed));
        packed_offset += block_packed;

        dequantize_block(block, out + start, count);
    }
}

}  // namespace quant
}  // namespace serius
