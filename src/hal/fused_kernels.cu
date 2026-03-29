#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_common.h"
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
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = is_max ? warp_reduce_max(val) : warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    float res = (threadIdx.x < blockDim.x / 32) ? shared[lane] : (is_max ? -1e20f : 0.0f);

    if (wid == 0) {
        res = is_max ? warp_reduce_max(res) : warp_reduce_sum(res);
        if (lane == 0) shared[0] = res;
    }
    __syncthreads();

    return (T)shared[0];
}

// --- LayerNorm + Residual Fusion ---

template <typename T>
__global__ void fused_layernorm_residual_kernel(T *x, const T *residual, const T *gamma, const T *beta, int n,
                                                float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

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
    if (!x || n == 0 || batch_seq == 0) return;
    dim3 grid((unsigned int)batch_seq);
    int threads = (n < 1024) ? ((int)(n + 31) / 32 * 32) : 1024;

    if (dtype == DType::Float16) {
        fused_layernorm_residual_kernel<<<grid, threads, 0, stream>>>(
            (__half *)x, (const __half *)residual, (const __half *)gamma, (const __half *)beta, (int)n, eps);
    } else if (dtype == DType::BFloat16) {
        fused_layernorm_residual_kernel<<<grid, threads, 0, stream>>>(
            (__nv_bfloat16 *)x, (const __nv_bfloat16 *)residual, (const __nv_bfloat16 *)gamma,
            (const __nv_bfloat16 *)beta, (int)n, eps);
    } else {
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

    int row_offset = (batch_head * seq_q + q_idx) * seq_k;
    int mask_offset = q_idx * seq_k;

    // 1. Find max with scale and mask
    float local_max = -1e30f;
    for (int i = tid; i < seq_k; i += blockDim.x) {
        bool is_masked = (mask) ? (mask[mask_offset + i] > 0.5f) : (i > q_idx);
        if (is_masked) continue;
        float val = to_float(x[row_offset + i]) * scale;
        local_max = fmaxf(local_max, val);
    }
    float max_val = reduce_block<float, true>(local_max);

    // 2. Compute sum of exps
    float exp_sum = 0.0f;
    for (int i = tid; i < seq_k; i += blockDim.x) {
        bool is_masked = (mask) ? (mask[mask_offset + i] > 0.5f) : (i > q_idx);
        if (is_masked) continue;
        float val = expf(to_float(x[row_offset + i]) * scale - max_val);
        exp_sum += val;
    }
    float total_sum = reduce_block<float, false>(exp_sum);

    // 3. Normalize
    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    for (int i = tid; i < seq_k; i += blockDim.x) {
        bool is_masked = (mask) ? (mask[mask_offset + i] > 0.5f) : (i > q_idx);
        if (is_masked) {
            x[row_offset + i] = (T)0.0f;
        } else {
            float val = expf(to_float(x[row_offset + i]) * scale - max_val);
            x[row_offset + i] = (T)(val * inv_sum);
        }
    }
}

void fused_masked_softmax(void *x, const void *mask, size_t batch_heads, size_t seq_q, size_t seq_k, DType dtype,
                          float scale, cudaStream_t stream)
{
    if (!x || batch_heads == 0 || seq_q == 0 || seq_k == 0) return;
    dim3 grid((unsigned int)batch_heads, (unsigned int)seq_q);
    int threads = (seq_k < 1024) ? ((int)(seq_k + 31) / 32 * 32) : 1024;

    if (dtype == DType::Float16) {
        fused_masked_softmax_kernel<<<grid, threads, 0, stream>>>((__half *)x, (const float *)mask, (int)batch_heads,
                                                                  (int)seq_q, (int)seq_k, scale);
    } else if (dtype == DType::BFloat16) {
        fused_masked_softmax_kernel<<<grid, threads, 0, stream>>>((__nv_bfloat16 *)x, (const float *)mask,
                                                                  (int)batch_heads, (int)seq_q, (int)seq_k, scale);
    } else {
        fused_masked_softmax_kernel<<<grid, threads, 0, stream>>>((float *)x, (const float *)mask, (int)batch_heads,
                                                                  (int)seq_q, (int)seq_k, scale);
    }
}

// --- Fused Bias + GELU ---

template <typename T>
__global__ void fused_bias_gelu_kernel(T *x, const T *bias, int n, int bias_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = to_float(x[idx]) + to_float(bias[idx % bias_size]);
        float gelu = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        x[idx] = (T)gelu;
    }
}

void fused_bias_gelu(void *x, const void *bias, size_t n, size_t bias_size, DType dtype, cudaStream_t stream)
{
    if (!x || !bias || n == 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    if (dtype == DType::Float16) {
        fused_bias_gelu_kernel<<<blocks, threads, 0, stream>>>((__half *)x, (const __half *)bias, (int)n, (int)bias_size);
    } else if (dtype == DType::BFloat16) {
        fused_bias_gelu_kernel<<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)x, (const __nv_bfloat16 *)bias,
                                                               (int)n, (int)bias_size);
    } else {
        fused_bias_gelu_kernel<<<blocks, threads, 0, stream>>>((float *)x, (const float *)bias, (int)n, (int)bias_size);
    }
}

// --- RMSNorm ---

template <typename T>
__global__ void rmsnorm_kernel(T *out, const T *input, const T *weight, int n, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = to_float(input[row * n + i]);
        sum_sq += val * val;
    }

    float block_sum_sq = reduce_block<float, false>(sum_sq);
    float inv_rms = rsqrtf(block_sum_sq / (float)n + eps);

    for (int i = tid; i < n; i += blockDim.x) {
        out[row * n + i] = (T)(to_float(input[row * n + i]) * inv_rms * to_float(weight[i]));
    }
}

void rmsnorm(void *out, const void *input, const void *weight, size_t n, size_t batch_seq, float eps, DType dtype,
              cudaStream_t stream)
{
    if (!out || !input || !weight || n == 0 || batch_seq == 0) return;
    dim3 grid((unsigned int)batch_seq);
    int threads = (n < 1024) ? ((int)(n + 31) / 32 * 32) : 1024;

    if (dtype == DType::Float16) {
        rmsnorm_kernel<<<grid, threads, 0, stream>>>((__half *)out, (const __half *)input, (const __half *)weight,
                                                     (int)n, eps);
    } else if (dtype == DType::BFloat16) {
        rmsnorm_kernel<<<grid, threads, 0, stream>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)input,
                                                     (const __nv_bfloat16 *)weight, (int)n, eps);
    } else {
        rmsnorm_kernel<<<grid, threads, 0, stream>>>((float *)out, (const float *)input, (const float *)weight, (int)n,
                                                     eps);
    }
}

// --- SwiGLU ---

template <typename T>
__global__ void swiglu_kernel(T *out, const T *x, const T *y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float vx = to_float(x[idx]);
        float vy = to_float(y[idx]);
        float silu = vx / (1.0f + expf(-vx));
        out[idx] = (T)(silu * vy);
    }
}

void swiglu(void *out, const void *x, const void *y, size_t n, DType dtype, cudaStream_t stream)
{
    if (!out || !x || !y || n == 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    if (dtype == DType::Float16) {
        swiglu_kernel<<<blocks, threads, 0, stream>>>((__half *)out, (const __half *)x, (const __half *)y, (int)n);
    } else if (dtype == DType::BFloat16) {
        swiglu_kernel<<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)x,
                                                      (const __nv_bfloat16 *)y, (int)n);
    } else {
        swiglu_kernel<<<blocks, threads, 0, stream>>>((float *)out, (const float *)x, (const float *)y, (int)n);
    }
}

// --- RoPE ---

template <typename T>
__global__ void rope_kernel(T *data, int head_dim, int n_heads, float base, int pos_offset)
{
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int i = threadIdx.x * 2;

    if (i < head_dim) {
        float pos = (float)(seq_idx + pos_offset);
        float freq = 1.0f / powf(base, (float)i / (float)head_dim);
        float theta = pos * freq;
        float cos_v = cosf(theta);
        float sin_v = sinf(theta);

        int offset = (seq_idx * n_heads * head_dim) + (head_idx * head_dim) + i;
        float x0 = to_float(data[offset]);
        float x1 = to_float(data[offset + 1]);

        data[offset] = (T)(x0 * cos_v - x1 * sin_v);
        data[offset + 1] = (T)(x0 * sin_v + x1 * cos_v);
    }
}

void rope(void *data, size_t head_dim, size_t seq_len, size_t n_heads, float base, int pos_offset, DType dtype,
          cudaStream_t stream)
{
    if (!data || head_dim == 0 || seq_len == 0 || n_heads == 0) return;

    dim3 grid((unsigned int)n_heads, (unsigned int)seq_len);
    int threads = (int)((head_dim / 2 + 31) / 32 * 32);
    if (threads > 1024) threads = 1024;

    if (dtype == DType::Float16) {
        rope_kernel<<<grid, threads, 0, stream>>>((__half *)data, (int)head_dim, (int)n_heads, base, pos_offset);
    } else if (dtype == DType::BFloat16) {
        rope_kernel<<<grid, threads, 0, stream>>>((__nv_bfloat16 *)data, (int)head_dim, (int)n_heads, base, pos_offset);
    } else {
        rope_kernel<<<grid, threads, 0, stream>>>((float *)data, (int)head_dim, (int)n_heads, base, pos_offset);
    }
}

// --- Softmax ---

template <typename T>
__global__ void softmax_kernel(T *x, int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_max = -1e20f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, to_float(x[row * cols + i]));
    }
    float max_val = reduce_block<float, true>(local_max);

    float exp_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(to_float(x[row * cols + i]) - max_val);
        x[row * cols + i] = (T)val;
        exp_sum += val;
    }
    float block_exp_sum = reduce_block<float, false>(exp_sum);
    float inv_sum = 1.0f / (block_exp_sum + 1e-9f);

    for (int i = tid; i < cols; i += blockDim.x) {
        x[row * cols + i] = (T)(to_float(x[row * cols + i]) * inv_sum);
    }
}

void softmax(void *x, size_t rows, size_t cols, DType dtype, cudaStream_t stream)
{
    if (!x || rows == 0 || cols == 0) return;
    dim3 grid((unsigned int)rows);
    int threads = (cols < 1024) ? ((int)(cols + 31) / 32 * 32) : 1024;

    if (dtype == DType::Float16) {
        softmax_kernel<<<grid, threads, 0, stream>>>((__half *)x, (int)rows, (int)cols);
    } else if (dtype == DType::BFloat16) {
        softmax_kernel<<<grid, threads, 0, stream>>>((__nv_bfloat16 *)x, (int)rows, (int)cols);
    } else {
        softmax_kernel<<<grid, threads, 0, stream>>>((float *)x, (int)rows, (int)cols);
    }
}

// --- Utility Kernels ---

template <typename T>
__global__ void add_inplace_kernel(T *out, const T *other, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (T)(to_float(out[i]) + to_float(other[i]));
    }
}

void add_inplace(void *out, const void *other, size_t n, DType dtype, cudaStream_t stream)
{
    if (!out || !other || n == 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    if (dtype == DType::Float32) {
        add_inplace_kernel<<<blocks, threads, 0, stream>>>((float *)out, (const float *)other, (int)n);
    } else if (dtype == DType::Float16) {
        add_inplace_kernel<<<blocks, threads, 0, stream>>>((__half *)out, (const __half *)other, (int)n);
    } else if (dtype == DType::BFloat16) {
        add_inplace_kernel<<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)other, (int)n);
    }
}

template <typename T>
__global__ void gather_kernel(T *out, const T *table, const int *indices, int n, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * dim) {
        int row = i / dim;
        int col = i % dim;
        int idx = indices[row];
        out[i] = table[idx * dim + col];
    }
}

void gather(void *out, const void *table, const int *indices, size_t n, size_t dim, DType dtype, cudaStream_t stream)
{
    if (!out || !table || !indices || n == 0 || dim == 0) return;
    int threads = 256;
    int blocks = (int)((n * dim + threads - 1) / threads);

    if (dtype == DType::Float32) {
        gather_kernel<<<blocks, threads, 0, stream>>>((float *)out, (const float *)table, indices, (int)n, (int)dim);
    } else if (dtype == DType::Float16) {
        gather_kernel<<<blocks, threads, 0, stream>>>((__half *)out, (const __half *)table, indices, (int)n, (int)dim);
    } else if (dtype == DType::BFloat16) {
        gather_kernel<<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)table, indices, (int)n, (int)dim);
    }
}


template <typename T>
__global__ void repeat_kv_kernel(T *out, const T *x, int batch_seq, int n_heads, int n_kv, int head_dim)
{
    int row = blockIdx.y; // batch_seq
    int head = blockIdx.x; // n_heads
    int tid = threadIdx.x;
    
    if (row < batch_seq && head < n_heads && tid < head_dim) {
        int rep = n_heads / n_kv;
        int src_head = head / rep;
        
        int out_idx = (row * n_heads + head) * head_dim + tid;
        int src_idx = (row * n_kv + src_head) * head_dim + tid;
        
        out[out_idx] = x[src_idx];
    }
}

void repeat_kv(void *out, const void *x, size_t batch_seq, size_t n_heads, size_t n_kv, size_t head_dim, 
               DType dtype, cudaStream_t stream)
{
    if (!out || !x || n_heads == 0 || head_dim == 0) return;
    
    dim3 grid((unsigned int)n_heads, (unsigned int)batch_seq);
    int threads = (head_dim < 1024) ? (int)head_dim : 1024;
    
    if (dtype == DType::Float16) {
        repeat_kv_kernel<<<grid, threads, 0, stream>>>((__half *)out, (const __half *)x, (int)batch_seq, (int)n_heads, (int)n_kv, (int)head_dim);
    } else if (dtype == DType::BFloat16) {
        repeat_kv_kernel<<<grid, threads, 0, stream>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)x, (int)batch_seq, (int)n_heads, (int)n_kv, (int)head_dim);
    } else {
        repeat_kv_kernel<<<grid, threads, 0, stream>>>((float *)out, (const float *)x, (int)batch_seq, (int)n_heads, (int)n_kv, (int)head_dim);
    }
}

template <typename T>
__global__ void copy_layout_kernel(T *dst, const T *src, size_t n, int dims, 
                                   const size_t *shape, const size_t *src_strides)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t src_idx = 0;
        size_t remaining = i;
        for (int d = dims - 1; d >= 0; --d) {
            size_t coord = remaining % shape[d];
            src_idx += coord * src_strides[d];
            remaining /= shape[d];
        }
        dst[i] = src[src_idx];
    }
}

void copy_layout(void *dst, const void *src, const size_t *shape, const size_t *src_strides, size_t dims,
                 DType dtype, cudaStream_t stream)
{
    if (!dst || !src || dims == 0) return;
    
    size_t n = 1;
    for (size_t i = 0; i < dims; ++i) n *= shape[i];
    if (n == 0) return;

    // We must copy shape and strides to GPU memory to be accessible by the kernel
    // For a small number of dims, we could pass them as arguments, but this is more general.
    size_t *d_shape = nullptr;
    size_t *d_strides = nullptr;
    CHECK_CUDA(cudaMalloc(&d_shape, dims * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_strides, dims * sizeof(size_t)));
    CHECK_CUDA(cudaMemcpyAsync(d_shape, shape, dims * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_strides, src_strides, dims * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    if (dtype == DType::Float32) {
        copy_layout_kernel<<<blocks, threads, 0, stream>>>((float *)dst, (const float *)src, n, (int)dims, d_shape, d_strides);
    } else if (dtype == DType::Float16) {
        copy_layout_kernel<<<blocks, threads, 0, stream>>>((__half *)dst, (const __half *)src, n, (int)dims, d_shape, d_strides);
    } else if (dtype == DType::BFloat16) {
        copy_layout_kernel<<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)dst, (const __nv_bfloat16 *)src, n, (int)dims, d_shape, d_strides);
    }

    // Cleanup device pointers after kernel launch (can be async if we had a proper stream-aware allocator)
    // For simplicity, we sync here, but in a production version we'd use a pool or an event.
    cudaStreamSynchronize(stream);
    cudaFree(d_shape);
    cudaFree(d_strides);
}

}  // namespace hal
}  // namespace serius
