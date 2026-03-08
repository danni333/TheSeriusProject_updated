#pragma once

#include <cstdint>
#include <vector>

#include "../hal/device.h"


namespace serius {
namespace quant {

// Conversions for a single value
uint16_t float_to_half(float f);
float half_to_float(uint16_t h);

uint16_t float_to_bf16(float f);
float bf16_to_float(uint16_t b);

// Bulk conversions
void convert_float_to_half(const float *src, uint16_t *dst, size_t n);
void convert_half_to_float(const uint16_t *src, float *dst, size_t n);

void convert_float_to_bf16(const float *src, uint16_t *dst, size_t n);
void convert_bf16_to_float(const uint16_t *src, float *dst, size_t n);

}  // namespace quant
}  // namespace serius
