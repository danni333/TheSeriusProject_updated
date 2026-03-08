#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "fused_kernels.h"

namespace cg = cooperative_groups;

namespace serius {
namespace hal {

// Helper to convert to float for calculation
__device__ __forceinline__ float atomicMax_float(float *addr, float val)
{
    int *addr_as_int = (int *)addr;
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template <typename T>
__device__ __forceinline__ float to_float(T val)
{
    return (float)val;
}

template <>
__device__ __forceinline__ float to_float(__half val)
{
    return __half2float(val);
}

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

template <typename T, bool is_max>
__device__ __forceinline__ T reduce_block(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = is_max ? warp_reduce_max(val) : warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : (is_max ? -1e20f : 0.0f);

    if (wid == 0) {
        val = is_max ? warp_reduce_max(val) : warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();

    return shared[0];
}

// --- LayerNorm + Residual Fusion ---

template <typename T>
__global__ void fused_layernorm_residual_kernel(T *x, const T *residual, const T *gamma, const T *beta, int n,
                                                float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    cg::thread_block block = cg::this_thread_block();

    // 1. Add residual and compute mean
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = to_float(x[row * n + i]);
        if (residual) val += to_float(residual[row * n + i]);
        x[row * n + i] = (T)val;
        sum += val;
    }

    float block_sum = reduce_block<float, false>(sum);
    float mean = block_sum / n;

    // 2. Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = to_float(x[row * n + i]) - mean;
        var_sum += diff * diff;
    }
    float block_var_sum = reduce_block<float, false>(var_sum);
    float inv_std = rsqrtf(block_var_sum / n + eps);

    // 3. Normalize and apply gamma/beta
    for (int i = tid; i < n; i += blockDim.x) {
        float val = to_float(x[row * n + i]);
        x[row * n + i] = (T)((val - mean) * inv_std * to_float(gamma[i]) + to_float(beta[i]));
    }
}

void fused_layernorm_residual(void *x, const void *residual, const void *gamma, const void *beta, size_t n,
                              size_t batch_seq, float eps, DType dtype, cudaStream_t stream)
{
    dim3 grid((unsigned int)batch_seq);
    int threads = (n < 1024) ? ((int)(n + 31) / 32 * 32) : 1024;

    if (dtype == DType::Float16) {
        fused_layernorm_residual_kernel<<<grid, threads, 0, stream>>>(
            (__half *)x, (const __half *)residual, (const __half *)gamma, (const __half *)beta, (int)n, eps);
    } else if (dtype == DType::BFloat16) {
        fused_layernorm_residual_kernel<<<grid, threads, 0, stream>>>(
            (__nv_bfloat16 *)x, (const __nv_bfloat16 *)residual, (const __nv_bfloat16 *)gamma,
            (const __nv_bfloat16 *)beta, (int)n, eps);
    } else {  // DType::Float32
        fused_layernorm_residual_kernel<<<grid, threads, 0, stream>>>(
            (float *)x, (const float *)residual, (const float *)gamma, (const float *)beta, (int)n, eps);
    }
}

// --- Fused Masked Softmax ---

template <typename T>
__global__ void fused_masked_softmax_kernel(T *x, const float *mask, int batch_heads, int seq_q, int seq_k, float scale)
{
    int batch_head = blockIdx.x;
    int q_idx = blockIdx.y;
    int tid = threadIdx.x;
    cg::thread_block block = cg::this_thread_block();

    int row_offset = (batch_head * seq_q + q_idx) * seq_k;
    int mask_offset = q_idx * seq_k;

    // 1. Find max with scale and mask
    float local_max = -1e20f;
    for (int i = tid; i < seq_k; i += blockDim.x) {
        float val = to_float(x[row_offset + i]) * scale;
        if (mask && mask[mask_offset + i] > 0.5f) {
            val = -1e20f;
        }
        local_max = fmaxf(local_max, val);
    }

    float max_val = reduce_block<float, true>(local_max);

    // 2. Compute sum of exps
    float exp_sum = 0.0f;
    for (int i = tid; i < seq_k; i += blockDim.x) {
        float val = to_float(x[row_offset + i]) * scale;
        if (mask && mask[mask_offset + i] > 0.5f) {
            val = 0.0f;
        } else {
            val = expf(val - max_val);
        }
        exp_sum += val;
        x[row_offset + i] = (T)val;
    }

    float block_exp_sum = reduce_block<float, false>(exp_sum);
    float inv_sum = 1.0f / (block_exp_sum + 1e-9f);

    // 3. Normalize
    for (int i = tid; i < seq_k; i += blockDim.x) {
        x[row_offset + i] = (T)(to_float(x[row_offset + i]) * inv_sum);
    }
}

void fused_masked_softmax(void *x, const void *mask, size_t batch_heads, size_t seq_q, size_t seq_k, DType dtype,
                          float scale, cudaStream_t stream)
{
    dim3 grid((unsigned int)batch_heads, (unsigned int)seq_q);
    int threads = 128;
    if (seq_k > 128) threads = 256;
    if (seq_k > 256) threads = 512;
    if (seq_k > 512) threads = 1024;

    if (dtype == DType::Float16) {
        fused_masked_softmax_kernel<<<grid, threads, 0, stream>>>((__half *)x, (const float *)mask, (int)batch_heads,
                                                                  (int)seq_q, (int)seq_k, scale);
    } else if (dtype == DType::BFloat16) {
        fused_masked_softmax_kernel<<<grid, threads, 0, stream>>>((__nv_bfloat16 *)x, (const float *)mask,
                                                                  (int)batch_heads, (int)seq_q, (int)seq_k, scale);
    } else {  // DType::Float32
        fused_masked_softmax_kernel<<<grid, threads, 0, stream>>>((float *)x, (const float *)mask, (int)batch_heads,
                                                                  (int)seq_q, (int)seq_k, scale);
    }
}

}  // namespace hal
}  // namespace serius
