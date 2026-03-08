#include "conversions.h"

#include <algorithm>
#include <cstring>


namespace serius {
namespace quant {

// FP32 to FP16 (IEEE 754)
uint16_t float_to_half(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);

    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mantissa = x & 0x7FFFFF;

    if (exp == 0) {
        // Zero or subnormal
        return static_cast<uint16_t>(sign);
    } else if (exp == 255) {
        // Infinity or NaN
        return static_cast<uint16_t>(sign | 0x7C00 | (mantissa ? 0x0200 : 0));
    } else {
        // Normalized
        int new_exp = static_cast<int>(exp) - 127 + 15;
        if (new_exp >= 31) {
            // Overflow to infinity
            return static_cast<uint16_t>(sign | 0x7C00);
        } else if (new_exp <= 0) {
            // Underflow to zero
            return static_cast<uint16_t>(sign);
        } else {
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(new_exp) << 10) | (mantissa >> 13));
        }
    }
}

// FP16 to FP32
float half_to_float(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    uint32_t exp = (static_cast<uint32_t>(h) & 0x7C00) >> 10;
    uint32_t mantissa = (static_cast<uint32_t>(h) & 0x03FF);

    uint32_t f_bits;
    if (exp == 0) {
        if (mantissa == 0) {
            // Zero
            f_bits = sign;
        } else {
            // Subnormal in FP16 becomes normalized in FP32
            while (!(mantissa & 0x0400)) {
                mantissa <<= 1;
                exp--;
            }
            exp++;
            mantissa &= ~0x0400;
            f_bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        // Infinity or NaN
        f_bits = sign | 0x7F800000 | (mantissa << 13);
    } else {
        // Normalized
        f_bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }

    float f;
    std::memcpy(&f, &f_bits, 4);
    return f;
}

// FP32 to BF16 (Simple truncation)
uint16_t float_to_bf16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    // Round to nearest even
    uint32_t lsb = (x >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    x += rounding_bias;
    return static_cast<uint16_t>(x >> 16);
}

// BF16 to FP32
float bf16_to_float(uint16_t b)
{
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

void convert_float_to_half(const float *src, uint16_t *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = float_to_half(src[i]);
    }
}

void convert_half_to_float(const uint16_t *src, float *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = half_to_float(src[i]);
    }
}

void convert_float_to_bf16(const float *src, uint16_t *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = float_to_bf16(src[i]);
    }
}

void convert_bf16_to_float(const uint16_t *src, float *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        dst[i] = bf16_to_float(src[i]);
    }
}

}  // namespace quant
}  // namespace serius
