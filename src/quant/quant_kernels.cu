#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cstdint>
#include <cmath>

namespace serius {
namespace quant {
namespace cuda {

// --- Constants ---
static __device__ const float E4M3_MAX_VAL = 448.0f;
static __device__ const float E5M2_MAX_VAL = 57344.0f;

// --- Device Helpers ---

__device__ uint8_t float_to_fp8_e4m3_dev(float f) {
    uint32_t bits = __float_as_uint(f);
    uint32_t sign = (bits >> 31) & 1;
    float abs_f = fabsf(f);

    if (abs_f == 0.0f) return (uint8_t)(sign << 7);
    if (abs_f > E4M3_MAX_VAL) return (uint8_t)((sign << 7) | 0x7E);

    int exp;
    float mant = frexpf(abs_f, &exp);
    mant *= 2.0f;
    exp -= 1;

    int biased_exp = exp + 7;
    if (biased_exp <= 0) {
        int shift = 1 - biased_exp;
        if (shift > 3) return (uint8_t)(sign << 7);
        uint32_t m = (uint32_t)((mant * 8.0f) + 0.5f);
        m >>= shift;
        if (m >= 8) return (uint8_t)((sign << 7) | 0x08);  // promote to smallest normal
        return (uint8_t)((sign << 7) | (m & 0x7));
    }

    uint32_t m = (uint32_t)(((mant - 1.0f) * 8.0f) + 0.5f);
    if (m >= 8) { m = 0; biased_exp++; }
    if (biased_exp >= 15) return (uint8_t)((sign << 7) | 0x7E);
    
    return (uint8_t)((sign << 7) | (biased_exp << 3) | (m & 0x7));
}

__device__ float fp8_e4m3_to_float_dev(uint8_t v) {
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp = (v >> 3) & 0xF;
    uint32_t man = v & 0x7;
    if (exp == 15 && man == 7) return nanf("");
    float res;
    if (exp == 0) res = (float)man * (1.0f / 512.0f);
    else res = (1.0f + (float)man / 8.0f) * __powf(2.0f, (float)exp - 7.0f);
    return sign ? -res : res;
}

__device__ uint8_t float_to_fp8_e5m2_dev(float f) {
    uint32_t bits = __float_as_uint(f);
    uint32_t sign = (bits >> 31) & 1;
    float abs_f = fabsf(f);

    if (abs_f == 0.0f) return (uint8_t)(sign << 7);
    if (abs_f > E5M2_MAX_VAL) return (uint8_t)((sign << 7) | 0x7C); // Infinity

    int exp;
    float mant = frexpf(abs_f, &exp);
    mant *= 2.0f;
    exp -= 1;

    int biased_exp = exp + 15;
    if (biased_exp <= 0) {
        int shift = 1 - biased_exp;
        if (shift > 2) return (uint8_t)(sign << 7);
        uint32_t m = (uint32_t)((mant * 4.0f) + 0.5f);
        m >>= shift;
        if (m >= 4) return (uint8_t)((sign << 7) | 0x04);  // promote to smallest normal
        return (uint8_t)((sign << 7) | (m & 0x3));
    }

    uint32_t m = (uint32_t)(((mant - 1.0f) * 4.0f) + 0.5f);
    if (m >= 4) { m = 0; biased_exp++; }
    if (biased_exp >= 31) return (uint8_t)((sign << 7) | 0x7C);
    
    return (uint8_t)((sign << 7) | (biased_exp << 2) | (m & 0x3));
}

__device__ float fp8_e5m2_to_float_dev(uint8_t v) {
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp = (v >> 2) & 0x1F;
    uint32_t man = v & 0x3;
    if (exp == 31) return (man == 0) ? (sign ? -huge_valf() : huge_valf()) : nanf("");
    float res;
    if (exp == 0) res = (float)man * (1.0f / 65536.0f);
    else res = (1.0f + (float)man / 4.0f) * __powf(2.0f, (float)exp - 15.0f);
    return sign ? -res : res;
}

// --- Kernels ---

__global__ void fp8_quantize_block_kernel(const float* data, uint8_t* out, float* scales, size_t n, size_t block_size, bool use_e4m3) {
    extern __shared__ float s_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    size_t start = (size_t)bid * block_size;
    
    float local_max = 0.0f;
    for (int i = tid; i < block_size; i += blockDim.x) {
        if (start + i < n) {
            local_max = fmaxf(local_max, fabsf(data[start + i]));
        }
    }
    
    s_mem[tid] = local_max;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_mem[tid] = fmaxf(s_mem[tid], s_mem[tid + s]);
        __syncthreads();
    }
    
    float amax = s_mem[0];
    float fp8_max = use_e4m3 ? E4M3_MAX_VAL : E5M2_MAX_VAL;
    float scale = (amax > 0.0f) ? (amax / fp8_max) : 1.0f;
    float inv_scale = 1.0f / scale;
    
    if (tid == 0) scales[bid] = scale;
    
    for (int i = tid; i < block_size; i += blockDim.x) {
        if (start + i < n) {
            float val = data[start + i] * inv_scale;
            out[start + i] = use_e4m3 ? float_to_fp8_e4m3_dev(val) : float_to_fp8_e5m2_dev(val);
        }
    }
}

__global__ void fp8_dequantize_block_kernel(const uint8_t* data, const float* scales, float* out, size_t n, size_t block_size, bool use_e4m3) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    size_t start = (size_t)bid * block_size;
    float scale = scales[bid];
    
    for (int i = tid; i < block_size; i += blockDim.x) {
        if (start + i < n) {
            float val = use_e4m3 ? fp8_e4m3_to_float_dev(data[start + i]) : fp8_e5m2_to_float_dev(data[start + i]);
            out[start + i] = val * scale;
        }
    }
}

// --- API Implementation ---

void float_to_fp8_e4m3_cuda(const float* src, uint8_t* dst, size_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // (Reuse existing kernel or implement one specialized)
}

void fp8_quantize_tensor_cuda(const float* data, uint8_t* fp8_out, float* scales_out, size_t n, size_t block_size, bool use_e4m3) {
    int threads = 128; // Power of 2 for reduction
    int blocks = (n + block_size - 1) / block_size;
    size_t shared_size = threads * sizeof(float);
    fp8_quantize_block_kernel<<<blocks, threads, shared_size>>>(data, fp8_out, scales_out, n, block_size, use_e4m3);
}

void fp8_dequantize_tensor_cuda(const uint8_t* fp8_data, const float* scales, float* out, size_t n, size_t block_size, bool use_e4m3) {
    int threads = 256;
    int blocks = (n + block_size - 1) / block_size;
    fp8_dequantize_block_kernel<<<blocks, threads>>>(fp8_data, scales, out, n, block_size, use_e4m3);
}

} // namespace cuda
} // namespace quant
} // namespace serius
