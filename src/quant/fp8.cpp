#include "fp8.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "quant_kernels.h"
#include "../hal/device.h"

namespace serius {
namespace quant {

// ============================================================
// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa
//   bias = 7, max exponent = 15 (but 15 is reserved for NaN/special)
//   max normal = 2^(14-7) * (1 + 7/8) = 2^7 * 1.875 = 240
//   Actually for E4M3FN (NVIDIA spec): max = 448 (exp=15 is valid, no inf)
//   min subnormal = 2^(1-7) * 2^(-3) = 2^(-9) = 1/512
// ============================================================

static constexpr int E4M3_EXP_BITS = 4;
static constexpr int E4M3_MAN_BITS = 3;
static constexpr int E4M3_BIAS = 7;
static constexpr float E4M3_MAX = 448.0f;                 // 2^8 * 1.75
static constexpr float E4M3_MIN_SUBNORM = 1.0f / 512.0f;  // 2^(-9)

uint8_t float_to_fp8_e4m3(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, 4);

    uint32_t sign = (bits >> 31) & 1;
    int32_t fp32_exp = static_cast<int32_t>((bits >> 23) & 0xFF);
    uint32_t fp32_man = bits & 0x7FFFFF;

    // Handle special cases
    if (fp32_exp == 255) {
        // Inf or NaN -> E4M3 NaN encoding (0x7F for +NaN, 0xFF for -NaN)
        // In E4M3FN there is no infinity, only NaN at exp=15, man=7
        return static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    float abs_f = std::fabs(f);

    if (abs_f == 0.0f) {
        return static_cast<uint8_t>(sign << 7);
    }

    // Clamp to E4M3 range
    if (abs_f > E4M3_MAX) {
        // Return max representable value (exp=15, man=6 -> 448)
        // exp=15 man=7 is NaN, so max is exp=15, man=6
        return static_cast<uint8_t>((sign << 7) | 0x7E);
    }

    if (abs_f < E4M3_MIN_SUBNORM * 0.5f) {
        // Too small, round to zero
        return static_cast<uint8_t>(sign << 7);
    }

    // Compute unbiased exponent in E4M3 space
    int32_t unbiased_exp = fp32_exp - 127;
    int32_t e4m3_biased_exp = unbiased_exp + E4M3_BIAS;

    if (e4m3_biased_exp <= 0) {
        // Subnormal in E4M3
        // Shift mantissa right to encode as subnormal
        int shift = 1 - e4m3_biased_exp;
        // The implicit 1 bit + top 3 mantissa bits
        uint32_t full_man = (1 << E4M3_MAN_BITS) | (fp32_man >> (23 - E4M3_MAN_BITS));
        uint32_t subnorm_man;
        if (shift >= E4M3_MAN_BITS + 1) {
            subnorm_man = 0;
        } else {
            // Round to nearest even
            uint32_t shifted = full_man >> shift;
            uint32_t remainder = full_man & ((1u << shift) - 1);
            uint32_t halfway = 1u << (shift - 1);
            if (remainder > halfway || (remainder == halfway && (shifted & 1))) {
                shifted++;
            }
            subnorm_man = shifted;
        }
        if (subnorm_man == 0) {
            return static_cast<uint8_t>(sign << 7);
        }
        // Subnormal: exp field = 0
        return static_cast<uint8_t>((sign << 7) | (subnorm_man & 0x07));
    }

    // Normal case: round mantissa from 23 bits to 3 bits
    uint32_t man3 = fp32_man >> (23 - E4M3_MAN_BITS);
    uint32_t remainder_bits = fp32_man & ((1u << (23 - E4M3_MAN_BITS)) - 1);
    uint32_t halfway = 1u << (23 - E4M3_MAN_BITS - 1);

    // Round to nearest even
    if (remainder_bits > halfway || (remainder_bits == halfway && (man3 & 1))) {
        man3++;
        if (man3 >= (1u << E4M3_MAN_BITS)) {
            man3 = 0;
            e4m3_biased_exp++;
        }
    }

    // Check for overflow after rounding
    if (e4m3_biased_exp > 15) {
        return static_cast<uint8_t>((sign << 7) | 0x7E);  // max value
    }
    // exp=15, man=7 is NaN, clamp to exp=15, man=6
    if (e4m3_biased_exp == 15 && man3 == 7) {
        man3 = 6;
    }

    return static_cast<uint8_t>((sign << 7) | (static_cast<uint8_t>(e4m3_biased_exp) << E4M3_MAN_BITS) | (man3 & 0x07));
}

float fp8_e4m3_to_float(uint8_t v)
{
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp = (v >> E4M3_MAN_BITS) & 0x0F;
    uint32_t man = v & 0x07;

    // NaN check: exp=15, man=7
    if (exp == 15 && man == 7) {
        uint32_t nan_bits = (sign << 31) | 0x7FC00000;
        float result;
        std::memcpy(&result, &nan_bits, 4);
        return result;
    }

    float abs_val;
    if (exp == 0) {
        if (man == 0) {
            abs_val = 0.0f;
        } else {
            // Subnormal: value = man * 2^(1-bias-mantissa_bits) = man * 2^(-9)
            abs_val = static_cast<float>(man) * (1.0f / 512.0f);
        }
    } else {
        // Normal: value = (1 + man/8) * 2^(exp - bias)
        abs_val = (1.0f + static_cast<float>(man) / 8.0f) * std::ldexp(1.0f, static_cast<int>(exp) - E4M3_BIAS);
    }

    return sign ? -abs_val : abs_val;
}

// ============================================================
// FP8 E5M2: 1 sign + 5 exponent + 2 mantissa
//   bias = 15, follows IEEE 754 conventions (has inf/nan)
//   max normal = 2^(30-15) * (1 + 3/4) = 2^15 * 1.75 = 57344
//   min subnormal = 2^(1-15) * 2^(-2) = 2^(-16) = 1/65536
// ============================================================

static constexpr int E5M2_EXP_BITS = 5;
static constexpr int E5M2_MAN_BITS = 2;
static constexpr int E5M2_BIAS = 15;
static constexpr float E5M2_MAX = 57344.0f;  // 2^15 * 1.75

uint8_t float_to_fp8_e5m2(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, 4);

    uint32_t sign = (bits >> 31) & 1;
    int32_t fp32_exp = static_cast<int32_t>((bits >> 23) & 0xFF);
    uint32_t fp32_man = bits & 0x7FFFFF;

    // Handle special cases
    if (fp32_exp == 255) {
        if (fp32_man != 0) {
            // NaN
            return static_cast<uint8_t>((sign << 7) | 0x7F);
        }
        // Infinity
        return static_cast<uint8_t>((sign << 7) | 0x7C);
    }

    float abs_f = std::fabs(f);

    if (abs_f == 0.0f) {
        return static_cast<uint8_t>(sign << 7);
    }

    // Clamp to E5M2 range
    if (abs_f > E5M2_MAX) {
        // Return infinity
        return static_cast<uint8_t>((sign << 7) | 0x7C);
    }

    int32_t unbiased_exp = fp32_exp - 127;
    int32_t e5m2_biased_exp = unbiased_exp + E5M2_BIAS;

    if (e5m2_biased_exp <= 0) {
        // Subnormal in E5M2
        int shift = 1 - e5m2_biased_exp;
        uint32_t full_man = (1 << E5M2_MAN_BITS) | (fp32_man >> (23 - E5M2_MAN_BITS));
        uint32_t subnorm_man;
        if (shift >= E5M2_MAN_BITS + 1) {
            subnorm_man = 0;
        } else {
            uint32_t shifted = full_man >> shift;
            uint32_t remainder = full_man & ((1u << shift) - 1);
            uint32_t halfway_val = 1u << (shift - 1);
            if (remainder > halfway_val || (remainder == halfway_val && (shifted & 1))) {
                shifted++;
            }
            subnorm_man = shifted;
        }
        if (subnorm_man == 0) {
            return static_cast<uint8_t>(sign << 7);
        }
        return static_cast<uint8_t>((sign << 7) | (subnorm_man & 0x03));
    }

    // Normal case: round mantissa from 23 bits to 2 bits
    uint32_t man2 = fp32_man >> (23 - E5M2_MAN_BITS);
    uint32_t remainder_bits = fp32_man & ((1u << (23 - E5M2_MAN_BITS)) - 1);
    uint32_t halfway = 1u << (23 - E5M2_MAN_BITS - 1);

    if (remainder_bits > halfway || (remainder_bits == halfway && (man2 & 1))) {
        man2++;
        if (man2 >= (1u << E5M2_MAN_BITS)) {
            man2 = 0;
            e5m2_biased_exp++;
        }
    }

    // Check for overflow after rounding
    if (e5m2_biased_exp >= 31) {
        // Overflow to infinity
        return static_cast<uint8_t>((sign << 7) | 0x7C);
    }

    return static_cast<uint8_t>((sign << 7) | (static_cast<uint8_t>(e5m2_biased_exp) << E5M2_MAN_BITS) | (man2 & 0x03));
}

float fp8_e5m2_to_float(uint8_t v)
{
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp = (v >> E5M2_MAN_BITS) & 0x1F;
    uint32_t man = v & 0x03;

    if (exp == 31) {
        if (man != 0) {
            // NaN
            uint32_t nan_bits = (sign << 31) | 0x7FC00000;
            float result;
            std::memcpy(&result, &nan_bits, 4);
            return result;
        }
        // Infinity
        uint32_t inf_bits = (sign << 31) | 0x7F800000;
        float result;
        std::memcpy(&result, &inf_bits, 4);
        return result;
    }

    float abs_val;
    if (exp == 0) {
        if (man == 0) {
            abs_val = 0.0f;
        } else {
            // Subnormal: value = man * 2^(1-bias-mantissa_bits) = man * 2^(-16)
            abs_val = static_cast<float>(man) * (1.0f / 65536.0f);
        }
    } else {
        // Normal: value = (1 + man/4) * 2^(exp - bias)
        abs_val = (1.0f + static_cast<float>(man) / 4.0f) * std::ldexp(1.0f, static_cast<int>(exp) - E5M2_BIAS);
    }

    return sign ? -abs_val : abs_val;
}

// ---- Bulk Conversions ----

void convert_float_to_fp8_e4m3(const float *src, uint8_t *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = float_to_fp8_e4m3(src[i]);
    }
}

void convert_fp8_e4m3_to_float(const uint8_t *src, float *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fp8_e4m3_to_float(src[i]);
    }
}

void convert_float_to_fp8_e5m2(const float *src, uint8_t *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = float_to_fp8_e5m2(src[i]);
    }
}

void convert_fp8_e5m2_to_float(const uint8_t *src, float *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fp8_e5m2_to_float(src[i]);
    }
}

// ---- Block-wise Scaling ----

FP8Block fp8_quantize_block(const float *data, size_t count, bool use_e4m3)
{
    FP8Block block;
    block.is_e4m3 = use_e4m3;
    block.data.resize(count);

    // Find the absolute maximum in the block
    float amax = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float abs_val = std::fabs(data[i]);
        if (abs_val > amax) amax = abs_val;
    }

    // Compute scale factor to map the data range to the FP8 representable range
    float fp8_max = use_e4m3 ? E4M3_MAX : E5M2_MAX;

    if (amax == 0.0f) {
        block.scale = 1.0f;
        std::fill(block.data.begin(), block.data.end(), uint8_t(0));
        return block;
    }

    // scale = amax / fp8_max, so that data / scale fits in FP8 range
    block.scale = amax / fp8_max;

    float inv_scale = 1.0f / block.scale;

    if (use_e4m3) {
        for (size_t i = 0; i < count; ++i) {
            block.data[i] = float_to_fp8_e4m3(data[i] * inv_scale);
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            block.data[i] = float_to_fp8_e5m2(data[i] * inv_scale);
        }
    }

    return block;
}

void fp8_dequantize_block(const FP8Block &block, float *out)
{
    size_t count = block.data.size();
    if (block.is_e4m3) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = fp8_e4m3_to_float(block.data[i]) * block.scale;
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            out[i] = fp8_e5m2_to_float(block.data[i]) * block.scale;
        }
    }
}

void fp8_quantize_tensor(const float *data, size_t n, size_t block_size, bool use_e4m3, std::vector<float> &scales_out,
                         std::vector<uint8_t> &fp8_out)
{
    size_t num_blocks = (n + block_size - 1) / block_size;
    scales_out.resize(num_blocks);
    fp8_out.resize(n);

    float fp8_max = use_e4m3 ? E4M3_MAX : E5M2_MAX;

    for (size_t b = 0; b < num_blocks; ++b) {
        size_t start = b * block_size;
        size_t count = std::min(block_size, n - start);
        const float *block_data = data + start;

        // Find amax for this block
        float amax = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            float abs_val = std::fabs(block_data[i]);
            if (abs_val > amax) amax = abs_val;
        }

        float scale = (amax > 0.0f) ? (amax / fp8_max) : 1.0f;
        scales_out[b] = scale;
        float inv_scale = 1.0f / scale;

        uint8_t *out_ptr = fp8_out.data() + start;
        if (use_e4m3) {
            for (size_t i = 0; i < count; ++i) {
                out_ptr[i] = float_to_fp8_e4m3(block_data[i] * inv_scale);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out_ptr[i] = float_to_fp8_e5m2(block_data[i] * inv_scale);
            }
        }
    }
}

void fp8_dequantize_tensor(const uint8_t *fp8_data, const float *scales, size_t n, size_t block_size, bool use_e4m3,
                           float *out)
{
    size_t num_blocks = (n + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; ++b) {
        size_t start = b * block_size;
        size_t count = std::min(block_size, n - start);
        float scale = scales[b];

        if (use_e4m3) {
            for (size_t i = 0; i < count; ++i) {
                out[start + i] = fp8_e4m3_to_float(fp8_data[start + i]) * scale;
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[start + i] = fp8_e5m2_to_float(fp8_data[start + i]) * scale;
            }
        }
    }
}

}  // namespace quant
}  // namespace serius
