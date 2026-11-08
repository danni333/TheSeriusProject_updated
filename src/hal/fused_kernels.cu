#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <math_constants.h>
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

// --- Logits GEMV (x @ W^T with W stored row-major [vocab, hidden]) ---
// One block per vocab row: coalesced reads of W, fp32 accumulation.
// x: [seq, hidden] fp32, w_t: [vocab, hidden] fp32, out: [seq, vocab] fp32.

__global__ void logits_gemv_kernel(const float *x, const float *w_t, float *out, int seq, int hidden)
{
    const int v = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float w_sh[4096];
    if (hidden > 4096) return;

    for (int i = tid; i < hidden; i += blockDim.x) {
        w_sh[i] = w_t[(size_t)v * hidden + i];
    }
    __syncthreads();

    const int vocab = gridDim.x;
    for (int s = 0; s < seq; ++s) {
        const float *xr = x + (size_t)s * hidden;
        float acc = 0.0f;
        for (int i = tid; i < hidden; i += blockDim.x) {
            acc += xr[i] * w_sh[i];
        }
        acc = reduce_block<float, false>(acc);
        if (tid == 0) {
            out[(size_t)s * vocab + v] = acc;
        }
        __syncthreads();
    }
}

void logits_gemv_f32(const void *x, const void *w_t, void *out, size_t seq, size_t vocab, size_t hidden,
                     cudaStream_t stream)
{
    if (!x || !w_t || !out || seq == 0 || vocab == 0 || hidden == 0 || hidden > 4096) return;
    logits_gemv_kernel<<<(int)vocab, 256, 0, stream>>>(
        (const float *)x, (const float *)w_t, (float *)out, (int)seq, (int)hidden);
}

// --- In-place scalar multiply ---

template <typename T>
__global__ void scale_inplace_kernel(T *data, int n, float s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = (T)(to_float(data[i]) * s);
    }
}

void scale_inplace(void *data, size_t n, float s, DType dtype, cudaStream_t stream)
{
    if (!data || n == 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    if (dtype == DType::Float32) {
        scale_inplace_kernel<<<blocks, threads, 0, stream>>>((float *)data, (int)n, s);
    } else if (dtype == DType::Float16) {
        scale_inplace_kernel<<<blocks, threads, 0, stream>>>((__half *)data, (int)n, s);
    } else if (dtype == DType::BFloat16) {
        scale_inplace_kernel<<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)data, (int)n, s);
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

// --- Mixed-precision helpers (FP32 residual stream + reduced-precision weights) ---

template <typename TIn, typename TOut, typename TW>
__global__ void rmsnorm_mixed_kernel(TOut *out, const TIn *input, const TW *weight, int n, float eps)
{
    int tid = threadIdx.x;
    int row = blockIdx.y;
    const TIn *in_row = input + (size_t)row * n;
    TOut *out_row = out + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = to_float(in_row[i]);
        sum_sq += v * v;
    }
    float block_sum_sq = reduce_block<float, false>(sum_sq);
    float inv_rms = rsqrtf(block_sum_sq / (float)n + eps);

    for (int i = tid; i < n; i += blockDim.x) {
        float v = to_float(in_row[i]) * inv_rms;
        float w = to_float(weight[i]);
        out_row[i] = (TOut)(v * w);
    }
}

void rmsnorm_mixed(void *out, const void *input, const void *weight, size_t n, size_t batch_seq, float eps,
                   DType out_dtype, cudaStream_t stream)
{
    // input is always FP32; weight and output share out_dtype.
    if (!out || !input || !weight || n == 0 || batch_seq == 0) return;
    dim3 grid(1, (unsigned int)batch_seq);
    int threads = 256;

    if (out_dtype == DType::Float16) {
        rmsnorm_mixed_kernel<float, __half, __half><<<grid, threads, 0, stream>>>(
            (__half *)out, (const float *)input, (const __half *)weight, (int)n, eps);
    } else if (out_dtype == DType::BFloat16) {
        rmsnorm_mixed_kernel<float, __nv_bfloat16, __nv_bfloat16><<<grid, threads, 0, stream>>>(
            (__nv_bfloat16 *)out, (const float *)input, (const __nv_bfloat16 *)weight, (int)n, eps);
    } else if (out_dtype == DType::Float32) {
        rmsnorm_mixed_kernel<float, float, float><<<grid, threads, 0, stream>>>(
            (float *)out, (const float *)input, (const float *)weight, (int)n, eps);
    }
}

template <typename TTable, typename TOut>
__global__ void gather_mixed_kernel(TOut *out, const TTable *table, const int *indices, int n, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * dim) {
        int row = i / dim;
        int col = i % dim;
        int idx = indices[row];
        out[i] = (TOut)to_float(table[idx * dim + col]);
    }
}

void gather_mixed(void *out, const void *table, const int *indices, size_t n, size_t dim, DType table_dtype,
                  DType out_dtype, cudaStream_t stream)
{
    if (!out || !table || !indices || n == 0 || dim == 0) return;
    int threads = 256;
    int blocks = (int)((n * dim + threads - 1) / threads);

    if (out_dtype == DType::Float32) {
        if (table_dtype == DType::Float16) {
            gather_mixed_kernel<__half, float><<<blocks, threads, 0, stream>>>((float *)out, (const __half *)table, indices, (int)n, (int)dim);
        } else if (table_dtype == DType::BFloat16) {
            gather_mixed_kernel<__nv_bfloat16, float><<<blocks, threads, 0, stream>>>((float *)out, (const __nv_bfloat16 *)table, indices, (int)n, (int)dim);
        } else {
            gather_mixed_kernel<float, float><<<blocks, threads, 0, stream>>>((float *)out, (const float *)table, indices, (int)n, (int)dim);
        }
    }
}

template <typename TAcc, typename TIn>
__global__ void add_inplace_mixed_kernel(TAcc *out, const TIn *other, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (TAcc)(to_float(out[i]) + to_float(other[i]));
    }
}

void add_inplace_mixed(void *out, const void *other, size_t n, DType acc_dtype, DType in_dtype, cudaStream_t stream)
{
    if (!out || !other || n == 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    if (acc_dtype == DType::Float32 && in_dtype == DType::Float16) {
        add_inplace_mixed_kernel<float, __half><<<blocks, threads, 0, stream>>>((float *)out, (const __half *)other, (int)n);
    } else if (acc_dtype == DType::Float32 && in_dtype == DType::BFloat16) {
        add_inplace_mixed_kernel<float, __nv_bfloat16><<<blocks, threads, 0, stream>>>((float *)out, (const __nv_bfloat16 *)other, (int)n);
    } else if (acc_dtype == DType::Float32 && in_dtype == DType::Float32) {
        add_inplace_mixed_kernel<float, float><<<blocks, threads, 0, stream>>>((float *)out, (const float *)other, (int)n);
    }
}

// --- Device-side dtype conversion (avoids host round-trips) ---

template <typename TOut>
__device__ __forceinline__ TOut from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename TIn, typename TOut>
__global__ void convert_dtype_kernel(TOut *out, const TIn *in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = from_float<TOut>(to_float(in[i]));
}

template <typename TIn>
static void launch_convert_out(void *out, const void *in, size_t n, DType out_dtype, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (out_dtype == DType::Float32) {
        convert_dtype_kernel<TIn, float><<<blocks, threads, 0, stream>>>((float *)out, (const TIn *)in, (int)n);
    } else if (out_dtype == DType::Float16) {
        convert_dtype_kernel<TIn, __half><<<blocks, threads, 0, stream>>>((__half *)out, (const TIn *)in, (int)n);
    } else if (out_dtype == DType::BFloat16) {
        convert_dtype_kernel<TIn, __nv_bfloat16><<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)out, (const TIn *)in, (int)n);
    }
}

void convert_dtype(void *out, const void *in, size_t n, DType in_dtype, DType out_dtype, cudaStream_t stream)
{
    if (!out || !in || n == 0 || in_dtype == out_dtype) return;
    if (in_dtype == DType::Float32) {
        launch_convert_out<float>(out, in, n, out_dtype, stream);
    } else if (in_dtype == DType::Float16) {
        launch_convert_out<__half>(out, in, n, out_dtype, stream);
    } else if (in_dtype == DType::BFloat16) {
        launch_convert_out<__nv_bfloat16>(out, in, n, out_dtype, stream);
    }
}

// --- GEMV: y[1,N] = x[1,K] @ W[K,N] (row-major W = {in, out} layout) ---
// One thread per output column; the warp reads a contiguous 64B row segment
// per k-step, so the whole weight matrix streams through memory exactly once.

template <typename T>
__global__ void gemv_kn_kernel(const T *x, const T *w, T *y, int K, int N)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        xs[i] = to_float(x[i]);
    }
    __syncthreads();

    float acc = 0.0f;
    const T *wrow = w + col;
    for (int k = 0; k < K; ++k) {
        acc = fmaf(xs[k], to_float(wrow[(size_t)k * N]), acc);
    }
    y[col] = (T)acc;
}

template <typename T>
__global__ void gemv_kn_vec4_kernel(const T *x, const T *w, T *y, int K, int N4)
{
    // Each thread produces 4 consecutive outputs via float4-style vectorized
    // loads (requires N % 4 == 0 and 16B-aligned w rows).
    const int col4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (col4 >= N4) return;
    const int col = col4 * 4;

    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        xs[i] = to_float(x[i]);
    }
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const T *wrow = w + col;
    for (int k = 0; k < K; ++k) {
        const T *p = wrow + (size_t)k * (N4 * 4);
        float xk = xs[k];
        acc0 = fmaf(xk, to_float(p[0]), acc0);
        acc1 = fmaf(xk, to_float(p[1]), acc1);
        acc2 = fmaf(xk, to_float(p[2]), acc2);
        acc3 = fmaf(xk, to_float(p[3]), acc3);
    }
    y[col] = (T)acc0;
    y[col + 1] = (T)acc1;
    y[col + 2] = (T)acc2;
    y[col + 3] = (T)acc3;
}

// Split-K variant: grid.y partitions the K loop so small-N GEMVs still fill
// the GPU. Partial sums are combined with atomicAdd into an FP32 buffer.
template <typename T>
__global__ void gemv_kn_sk_kernel(const T *x, const T *w, float *y, int K, int N, int ksplit)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    const int chunk = (K + ksplit - 1) / ksplit;
    const int k0 = blockIdx.y * chunk;
    const int k1 = min(K, k0 + chunk);
    if (k0 >= K) return;

    float acc = 0.0f;
    const T *wrow = w + col;
    for (int k = k0; k < k1; ++k) {
        acc = fmaf(to_float(x[k]), to_float(wrow[(size_t)k * N]), acc);
    }
    atomicAdd(y + col, acc);
}

/// y[1,N] (FP16) = x[1,K] @ W[K,N] (FP16, row-major {in,out}) with FP32
/// accumulation via a zeroed scratch buffer + convert_dtype on the caller side.
void gemv_kn_sk_f16(const void *x, const void *w, void *y_f32_acc, size_t K, size_t N,
                    cudaStream_t stream)
{
    if (!x || !w || !y_f32_acc || K == 0 || N == 0) return;
    const int threads = 256;
    // Aim for a few hundred blocks: split K more aggressively for small N.
    const int ksplit = (N <= 2048) ? 16 : 8;
    dim3 grid((unsigned)((N + threads - 1) / threads), (unsigned)ksplit);
    gemv_kn_sk_kernel<__half><<<grid, threads, 0, stream>>>(
        (const __half *)x, (const __half *)w, (float *)y_f32_acc, (int)K, (int)N, ksplit);
}

void gemv_kn(void *x, const void *w, void *y, size_t K, size_t N, DType dtype, cudaStream_t stream)
{
    if (!x || !w || !y || K == 0 || N == 0) return;
    const int threads = 256;
    const size_t shmem = K * sizeof(float);

    if (dtype == DType::Float16) {
        if ((N & 3) == 0) {
            gemv_kn_vec4_kernel<__half><<<(unsigned)((N / 4 + threads - 1) / threads), threads, shmem, stream>>>(
                (const __half *)x, (const __half *)w, (__half *)y, (int)K, (int)(N / 4));
        } else {
            gemv_kn_kernel<__half><<<(unsigned)((N + threads - 1) / threads), threads, shmem, stream>>>(
                (const __half *)x, (const __half *)w, (__half *)y, (int)K, (int)N);
        }
    } else if (dtype == DType::BFloat16) {
        gemv_kn_kernel<__nv_bfloat16><<<(unsigned)((N + threads - 1) / threads), threads, shmem, stream>>>(
            (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)w, (__nv_bfloat16 *)y, (int)K, (int)N);
    } else if (dtype == DType::Float32) {
        gemv_kn_kernel<float><<<(unsigned)((N + threads - 1) / threads), threads, shmem, stream>>>(
            (const float *)x, (const float *)w, (float *)y, (int)K, (int)N);
    }
}

// --- Quantized GEMV (GGUF Q4_0 / Q4_1 / Q8_0) with on-the-fly dequant ---
// y[row] = sum_k x[k] * dequant(W_row). Split-K over 32-weight blocks with
// FP32 atomic accumulation into a zeroed scratch buffer.
// QuantType values mirror quant::QuantType (3=Q4_0, 4=Q4_1, 5=Q8_0).

template <int QT>
__global__ void gemv_quant_kernel(const __half *x, const uint8_t *w, float *y, int in, int out, int bpr,
                                  int ksplit)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out) return;
    const int chunk = (bpr + ksplit - 1) / ksplit;
    const int b0 = blockIdx.y * chunk;
    const int b1 = min(bpr, b0 + chunk);
    if (b0 >= bpr) return;

    constexpr int BB = (QT == 3) ? 18 : (QT == 4) ? 20 : 34;
    const uint8_t *wr = w + (size_t)row * bpr * BB;
    float acc = 0.0f;

    for (int b = b0; b < b1; ++b) {
        const uint8_t *blk = wr + (size_t)b * BB;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        const int base = b * 32;
        if (QT == 3) {  // Q4_0: w = (q - 8) * d; lo nibbles -> k..k+15, hi -> k+16..k+31
            const uint8_t *q = blk + 2;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                acc = fmaf((float)((q[j] & 0xF) - 8) * d, __half2float(x[base + j]), acc);
                acc = fmaf((float)((q[j] >> 4) - 8) * d, __half2float(x[base + j + 16]), acc);
            }
        } else if (QT == 4) {  // Q4_1: w = q * d + m (same nibble layout)
            const float m = __half2float(*reinterpret_cast<const __half *>(blk + 2));
            const uint8_t *q = blk + 4;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                acc = fmaf((float)(q[j] & 0xF) * d + m, __half2float(x[base + j]), acc);
                acc = fmaf((float)(q[j] >> 4) * d + m, __half2float(x[base + j + 16]), acc);
            }
        } else {  // Q8_0: w = q * d
            const int8_t *q = reinterpret_cast<const int8_t *>(blk + 2);
            #pragma unroll 4
            for (int j = 0; j < 32; ++j) {
                acc = fmaf((float)q[j] * d, __half2float(x[base + j]), acc);
            }
        }
    }
    atomicAdd(y + row, acc);
}

// Warp-per-row variant: each warp owns one output row, lanes stride over
// 32-weight blocks so packed bytes are read coalesced. x is FP32 (pre-converted
// once per projection input). Writes y[row] directly — no atomics/zeroing.
template <int QT>
__global__ void gemv_quant_warp_kernel(const float *x, const uint8_t *w, float *y, int in, int out,
                                       int bpr)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= out) return;

    constexpr int BB = (QT == 3) ? 18 : (QT == 4) ? 20 : 34;
    const uint8_t *wr = w + (size_t)warp * bpr * BB;
    float acc = 0.0f;

    for (int b = lane; b < bpr; b += 32) {
        const uint8_t *blk = wr + (size_t)b * BB;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        const int base = b * 32;
        if (QT == 3) {  // Q4_0: w = (q - 8) * d; lo nibbles -> k..k+15, hi -> k+16..k+31
            const uint8_t *q = blk + 2;
            float a0 = 0.0f, a1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                a0 = fmaf((float)((q[j] & 0xF) - 8) * d, x[base + j], a0);
                a1 = fmaf((float)((q[j] >> 4) - 8) * d, x[base + j + 16], a1);
            }
            acc += a0 + a1;
        } else if (QT == 4) {  // Q4_1: w = q * d + m (same nibble layout)
            const float m = __half2float(*reinterpret_cast<const __half *>(blk + 2));
            const uint8_t *q = blk + 4;
            float a0 = 0.0f, a1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                a0 = fmaf((float)(q[j] & 0xF) * d + m, x[base + j], a0);
                a1 = fmaf((float)(q[j] >> 4) * d + m, x[base + j + 16], a1);
            }
            acc += a0 + a1;
        } else {  // Q8_0: w = q * d
            const int8_t *q = reinterpret_cast<const int8_t *>(blk + 2);
            float a0 = 0.0f, a1 = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < 32; ++j) {
                if (j < 16) a0 = fmaf((float)q[j] * d, x[base + j], a0);
                else        a1 = fmaf((float)q[j] * d, x[base + j], a1);
            }
            acc += a0 + a1;
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    if (lane == 0) { y[warp] = acc; }
}

void gemv_quant_f32(const void *x_f32, const void *w, int qtype, size_t in, size_t out, void *y_f32,
                    cudaStream_t stream)
{
    if (!x_f32 || !w || !y_f32 || in == 0 || out == 0 || (in % 32) != 0) return;
    const int bpr = (int)(in / 32);
    const int threads = 256;                    // 8 warps -> 8 rows per block
    const unsigned rows_per_block = threads / 32;
    const unsigned bx = (unsigned)((out + rows_per_block - 1) / rows_per_block);
    dim3 grid(bx);

    if (qtype == 3) {
        gemv_quant_warp_kernel<3><<<grid, threads, 0, stream>>>((const float *)x_f32, (const uint8_t *)w,
                                                             (float *)y_f32, (int)in, (int)out, bpr);
    } else if (qtype == 4) {
        gemv_quant_warp_kernel<4><<<grid, threads, 0, stream>>>((const float *)x_f32, (const uint8_t *)w,
                                                             (float *)y_f32, (int)in, (int)out, bpr);
    } else if (qtype == 5) {
        gemv_quant_warp_kernel<5><<<grid, threads, 0, stream>>>((const float *)x_f32, (const uint8_t *)w,
                                                             (float *)y_f32, (int)in, (int)out, bpr);
    }
}

// Vectorized warp-per-row GEMV over the repacked device layout
// (QuantWeight::dev_bytes): each row is [bpr*qs quant bytes, block-major]
// [bpr fp16 d][bpr fp16 m (Q4_1)], padded to 16B. `stride` is the row stride,
// `d_off`/`m_off` are row-relative offsets of the scale arrays, `qs` is the
// per-block quant payload size (16 or 32). x is FP32, y FP32 written directly.
// kind: 3=Q4_0, 4=Q4_1, 5=Q8_0.
__device__ __forceinline__ uint32_t sel_u32(uint4 v, int j)
{
    switch (j & 3) {
        case 0: return v.x;
        case 1: return v.y;
        case 2: return v.z;
        default: return v.w;
    }
}

__global__ void gemv_quant_vec_kernel(const float *x, const float *sx16, const uint8_t *w, float *y,
                                      int in, int out, int stride, int qs, int d_off, int m_off, int kind,
                                      int rows_per_warp)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int r0 = warp * rows_per_warp;
    if (r0 >= out) return;

    const int bpr = in >> 5;  // in % 32 == 0 guaranteed by repack guard
    const bool has_m = (kind == 4);

    if (kind == 3 || kind == 4) {
        // Q4_0/Q4_1 fast path: float4 x loads, two accumulators per row.
        // rows_per_warp=4 gives each warp four independent weight streams,
        // maximizing memory-level parallelism (bench: +20% GB/s vs 2).
        constexpr int RPW = 4;
        float accA[RPW], accB[RPW];
        const uint8_t *wr[RPW], *darr[RPW], *marr[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            accA[r] = 0.0f; accB[r] = 0.0f;
            if (r0 + r < out) {
                wr[r] = w + (size_t)(r0 + r) * stride;
                darr[r] = wr[r] + d_off;
                marr[r] = (m_off != 0) ? wr[r] + m_off : nullptr;
            }
        }
        for (int b = lane; b < bpr; b += 32) {
            // __ldcs: weights are read exactly once per token — evict-first
            // keeps L2 for the reused x vector and scale arrays.
            const int base = b * 32;
            const float sx0 = __ldg(sx16 + 2 * b);
            const float sx1 = __ldg(sx16 + 2 * b + 1);
            #pragma unroll
            for (int r = 0; r < RPW; ++r) {
                if (r0 + r >= out) break;
                const uint4 v = __ldcs(reinterpret_cast<const uint4 *>(wr[r] + (size_t)b * qs));
                const float d = __half2float(*reinterpret_cast<const __half *>(darr[r] + b * 2));
                const float m = has_m ? __half2float(*reinterpret_cast<const __half *>(marr[r] + b * 2)) : 0.0f;
                // inline fold with shared sx (same b for all rows)
                const float4 *x4lo = reinterpret_cast<const float4 *>(x + base);
                const float4 *x4hi = reinterpret_cast<const float4 *>(x + base + 16);
                float ta = 0.0f, tb = 0.0f;
                #pragma unroll
                for (int j = 0; j < 16; j += 4) {
                    const uint32_t w0 = sel_u32(v, (j + 0) >> 2);
                    const uint32_t w1 = sel_u32(v, (j + 1) >> 2);
                    const uint32_t w2 = sel_u32(v, (j + 2) >> 2);
                    const uint32_t w3 = sel_u32(v, (j + 3) >> 2);
                    const uint32_t b0 = (w0 >> (((j + 0) & 3) * 8)) & 0xFFu;
                    const uint32_t b1 = (w1 >> (((j + 1) & 3) * 8)) & 0xFFu;
                    const uint32_t b2 = (w2 >> (((j + 2) & 3) * 8)) & 0xFFu;
                    const uint32_t b3 = (w3 >> (((j + 3) & 3) * 8)) & 0xFFu;
                    const float4 xl = __ldg(x4lo + (j >> 2));
                    const float4 xh = __ldg(x4hi + (j >> 2));
                    ta = fmaf((float)(int)(b0 & 0xFu), xl.x, ta);
                    tb = fmaf((float)(int)(b0 >> 4), xh.x, tb);
                    ta = fmaf((float)(int)(b1 & 0xFu), xl.y, ta);
                    tb = fmaf((float)(int)(b1 >> 4), xh.y, tb);
                    ta = fmaf((float)(int)(b2 & 0xFu), xl.z, ta);
                    tb = fmaf((float)(int)(b2 >> 4), xh.z, tb);
                    ta = fmaf((float)(int)(b3 & 0xFu), xl.w, ta);
                    tb = fmaf((float)(int)(b3 >> 4), xh.w, tb);
                }
                accA[r] = fmaf(d, ta, accA[r]);
                accB[r] = fmaf(d, tb, accB[r]);
                if (has_m) {
                    accA[r] = fmaf(m, sx0, accA[r]);
                    accB[r] = fmaf(m, sx1, accB[r]);
                } else {
                    accA[r] = fmaf(-8.0f * d, sx0, accA[r]);
                    accB[r] = fmaf(-8.0f * d, sx1, accB[r]);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (r0 + r >= out) break;
            float acc = accA[r] + accB[r];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, o);
            }
            if (lane == 0) { y[r0 + r] = acc; }
        }
        return;
    }

    // Generic path (Q8_0): one row per warp.
    const uint8_t *wr = w + (size_t)r0 * stride;
    const uint8_t *darr = wr + d_off;
    const uint8_t *marr = (m_off != 0) ? wr + m_off : nullptr;

    float acc = 0.0f;
    for (int b = lane; b < bpr; b += 32) {
        const uint4 *qp = reinterpret_cast<const uint4 *>(wr + (size_t)b * qs);
        const uint4 v0 = qp[0];
        const uint4 v1 = (qs > 16) ? qp[1] : make_uint4(0u, 0u, 0u, 0u);
        const float d = __half2float(*reinterpret_cast<const __half *>(darr + b * 2));
        const float m = has_m ? __half2float(*reinterpret_cast<const __half *>(marr + b * 2)) : 0.0f;
        const int base = b * 32;

        if (kind == 5) {  // Q8_0: 32 int8 weights
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const uint4 vv = (j < 16) ? v0 : v1;
                const int jb = j & 15;
                const uint32_t byte = (sel_u32(vv, jb >> 2) >> ((jb & 3) * 8)) & 0xFFu;
                acc = fmaf((float)(int8_t)byte * d, x[base + j], acc);
            }
        } else {          // Q4_0 / Q4_1: 16 bytes -> 32 weights
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint32_t byte = (sel_u32(v0, j >> 2) >> ((j & 3) * 8)) & 0xFFu;
                const float lo = (kind == 4) ? ((float)(int)(byte & 0xFu) * d + m)
                                             : ((float)((int)(byte & 0xFu) - 8) * d);
                const float hi = (kind == 4) ? ((float)(int)(byte >> 4) * d + m)
                                             : ((float)((int)(byte >> 4) - 8) * d);
                acc = fmaf(lo, x[base + j], acc);
                acc = fmaf(hi, x[base + j + 16], acc);
            }
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    if (lane == 0) { y[warp] = acc; }
}

// F16-IO variant: x is __half (converted in-register — FP16->FP32 is exact,
// so results are identical to the convert_dtype + f32-x path), y is written
// directly as __half or float. Eliminates two convert_dtype launches per
// projection. Bench: ~217 GB/s vs ~182 GB/s for the f32-x variant.
__device__ __forceinline__ float2 h22f2(uint32_t u)
{
    return __half22float2(*reinterpret_cast<const __half2 *>(&u));
}

__global__ void gemv_quant_vec_h_kernel(const __half *x, const float *sx16, const uint8_t *w,
                                        void *y, int y_f16, int in, int out, int stride, int qs,
                                        int d_off, int m_off, int kind, int rows_per_warp)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int r0 = warp * rows_per_warp;
    if (r0 >= out) return;

    const int bpr = in >> 5;
    const bool has_m = (kind == 4);

    if (kind == 3 || kind == 4) {
        // Q4_0/Q4_1 fast path, k-major lane mapping: each lane owns ONE row
        // (lane>>3, 8 contiguous lanes per row) and strides k-blocks by 8
        // (lane&7). The old mapping (all 32 lanes on k, 4 rows inner) wasted
        // ~33% of the second k-iteration whenever bpr%32 != 0 — which is the
        // common case (in=1536 -> bpr=48). This mapping is exactly balanced
        // for any bpr multiple of 8 and needs fewer bytes per lane.
        const int row_local = lane >> 3;          // 0..3
        const int q = lane & 7;                   // k-stride group
        const int rrow = r0 + row_local;
        // Clamp to a valid row so out-of-range lanes still participate in
        // the __shfl_xor_sync reduction (mask requires all 32 lanes).
        const int rclamp = (rrow < out) ? rrow : (out - 1);
        const uint8_t *wr = w + (size_t)rclamp * stride;
        const uint8_t *darr = wr + d_off;
        const uint8_t *marr = (m_off != 0) ? wr + m_off : nullptr;

        float accA = 0.0f, accB = 0.0f;
        const int maxit = (bpr + 7) >> 3;         // ceil(bpr/8)
        for (int t = 0; t < maxit; t += 4) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int b = q + 8 * (t + u);
                if (b >= bpr) continue;
                const uint4 v = __ldcs(reinterpret_cast<const uint4 *>(wr + (size_t)b * qs));
                const float d = __half2float(*reinterpret_cast<const __half *>(darr + b * 2));
                const float m = has_m ? __half2float(*reinterpret_cast<const __half *>(marr + b * 2)) : 0.0f;
                const float sx0 = __ldg(sx16 + 2 * b);
                const float sx1 = __ldg(sx16 + 2 * b + 1);
                const int base = b * 32;
                const __half *xl_p = x + base;
                const __half *xh_p = x + base + 16;
                float ta = 0.0f, tb = 0.0f;
                #pragma unroll
                for (int j = 0; j < 16; j += 4) {
                    const uint32_t w0 = sel_u32(v, (j + 0) >> 2);
                    const uint32_t w1 = sel_u32(v, (j + 1) >> 2);
                    const uint32_t w2 = sel_u32(v, (j + 2) >> 2);
                    const uint32_t w3 = sel_u32(v, (j + 3) >> 2);
                    const uint32_t b0 = (w0 >> (((j + 0) & 3) * 8)) & 0xFFu;
                    const uint32_t b1 = (w1 >> (((j + 1) & 3) * 8)) & 0xFFu;
                    const uint32_t b2 = (w2 >> (((j + 2) & 3) * 8)) & 0xFFu;
                    const uint32_t b3 = (w3 >> (((j + 3) & 3) * 8)) & 0xFFu;
                    const uint2 xl2 = __ldg(reinterpret_cast<const uint2 *>(xl_p + j));
                    const uint2 xh2 = __ldg(reinterpret_cast<const uint2 *>(xh_p + j));
                    const float2 xa = h22f2(xl2.x), xb = h22f2(xl2.y);
                    const float2 xc = h22f2(xh2.x), xd = h22f2(xh2.y);
                    ta = fmaf((float)(int)(b0 & 0xFu), xa.x, ta);
                    tb = fmaf((float)(int)(b0 >> 4), xc.x, tb);
                    ta = fmaf((float)(int)(b1 & 0xFu), xa.y, ta);
                    tb = fmaf((float)(int)(b1 >> 4), xc.y, tb);
                    ta = fmaf((float)(int)(b2 & 0xFu), xb.x, ta);
                    tb = fmaf((float)(int)(b2 >> 4), xd.x, tb);
                    ta = fmaf((float)(int)(b3 & 0xFu), xb.y, ta);
                    tb = fmaf((float)(int)(b3 >> 4), xd.y, tb);
                }
                accA = fmaf(d, ta, accA);
                accB = fmaf(d, tb, accB);
                if (has_m) {
                    accA = fmaf(m, sx0, accA);
                    accB = fmaf(m, sx1, accB);
                } else {
                    accA = fmaf(-8.0f * d, sx0, accA);
                    accB = fmaf(-8.0f * d, sx1, accB);
                }
            }
        }
        // Reduce across the 8 lanes sharing this row (contiguous lane group).
        float acc = accA + accB;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, o, 8);
        }
        if ((lane & 7) == 0 && rrow < out) {
            if (y_f16) reinterpret_cast<__half *>(y)[rrow] = __float2half(acc);
            else reinterpret_cast<float *>(y)[rrow] = acc;
        }
        return;
    }

    // Generic path (Q8_0): one row per warp.
    const uint8_t *wr = w + (size_t)r0 * stride;
    const uint8_t *darr = wr + d_off;
    const uint8_t *marr = (m_off != 0) ? wr + m_off : nullptr;

    float acc = 0.0f;
    for (int b = lane; b < bpr; b += 32) {
        const uint4 *qp = reinterpret_cast<const uint4 *>(wr + (size_t)b * qs);
        const uint4 v0 = qp[0];
        const uint4 v1 = (qs > 16) ? qp[1] : make_uint4(0u, 0u, 0u, 0u);
        const float d = __half2float(*reinterpret_cast<const __half *>(darr + b * 2));
        const float m = has_m ? __half2float(*reinterpret_cast<const __half *>(marr + b * 2)) : 0.0f;
        const int base = b * 32;

        if (kind == 5) {  // Q8_0: 32 int8 weights
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const uint4 vv = (j < 16) ? v0 : v1;
                const int jb = j & 15;
                const uint32_t byte = (sel_u32(vv, jb >> 2) >> ((jb & 3) * 8)) & 0xFFu;
                acc = fmaf((float)(int8_t)byte * d, __half2float(x[base + j]), acc);
            }
        } else {          // Q4_0 / Q4_1: 16 bytes -> 32 weights
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint32_t byte = (sel_u32(v0, j >> 2) >> ((j & 3) * 8)) & 0xFFu;
                const float lo = (kind == 4) ? ((float)(int)(byte & 0xFu) * d + m)
                                             : ((float)((int)(byte & 0xFu) - 8) * d);
                const float hi = (kind == 4) ? ((float)(int)(byte >> 4) * d + m)
                                             : ((float)((int)(byte >> 4) - 8) * d);
                acc = fmaf(lo, __half2float(x[base + j]), acc);
                acc = fmaf(hi, __half2float(x[base + j + 16]), acc);
            }
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    if (lane == 0) {
        if (y_f16) reinterpret_cast<__half *>(y)[warp] = __float2half(acc);
        else reinterpret_cast<float *>(y)[warp] = acc;
    }
}

__global__ void block_sums16_h_kernel(const __half *x, float *sx, int n16)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n16) return;
    const __half *p = x + i * 16;
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j += 4) {
        const uint2 v = __ldg(reinterpret_cast<const uint2 *>(p + j));
        const float2 a = h22f2(v.x), b = h22f2(v.y);
        s += (a.x + a.y) + (b.x + b.y);
    }
    sx[i] = s;
}

void gemv_quant_vec_f16io(const void *x_f16, const void *sx16, const void *w, int kind,
                         size_t in, size_t out, int stride, int qs, int d_off, int m_off,
                         void *y, bool y_f16, cudaStream_t stream)
{
    if (!x_f16 || !w || !y || in == 0 || out == 0 || (in % 32) != 0) return;
    if ((kind == 3 || kind == 4) && sx16 == nullptr) return;
    const int threads = 256;
    const unsigned warps_per_block = threads / 32;
    const int rows_per_warp = (kind == 3 || kind == 4) ? 4 : 1;
    const unsigned rows_per_block = warps_per_block * (unsigned)rows_per_warp;
    const unsigned bx = (unsigned)((out + rows_per_block - 1) / rows_per_block);
    gemv_quant_vec_h_kernel<<<bx, threads, 0, stream>>>(
        (const __half *)x_f16, (const float *)sx16, (const uint8_t *)w, y, y_f16 ? 1 : 0,
        (int)in, (int)out, stride, qs, d_off, m_off, kind, rows_per_warp);
}

void block_sums16_f16(const void *x_f16, void *sx16, int n16, cudaStream_t stream)
{
    if (!x_f16 || !sx16 || n16 <= 0) return;
    const int threads = 256;
    const unsigned bx = (unsigned)((n16 + threads - 1) / threads);
    block_sums16_h_kernel<<<bx, threads, 0, stream>>>((const __half *)x_f16, (float *)sx16, n16);
}

void gemv_quant_vec_f32(const void *x_f32, const void *sx16, const void *w, int kind,
                        size_t in, size_t out, int stride, int qs, int d_off, int m_off,
                        void *y_f32, cudaStream_t stream)
{
    if (!x_f32 || !w || !y_f32 || in == 0 || out == 0 || (in % 32) != 0) return;
    if ((kind == 3 || kind == 4) && sx16 == nullptr) return;
    const int threads = 256;
    const unsigned warps_per_block = threads / 32;
    // Q4_0/Q4_1: four rows per warp for more memory-level parallelism.
    const int rows_per_warp = (kind == 3 || kind == 4) ? 4 : 1;
    const unsigned rows_per_block = warps_per_block * (unsigned)rows_per_warp;
    const unsigned bx = (unsigned)((out + rows_per_block - 1) / rows_per_block);
    gemv_quant_vec_kernel<<<bx, threads, 0, stream>>>(
        (const float *)x_f32, (const float *)sx16, (const uint8_t *)w, (float *)y_f32,
        (int)in, (int)out, stride, qs, d_off, m_off, kind, rows_per_warp);
}

__global__ void block_sums16_kernel(const float *x, float *sx, int n16)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n16) return;
    const float4 *x4 = reinterpret_cast<const float4 *>(x + i * 16);
    const float4 a = __ldg(x4);
    const float4 b = __ldg(x4 + 1);
    const float4 c = __ldg(x4 + 2);
    const float4 d = __ldg(x4 + 3);
    sx[i] = (a.x + a.y + a.z + a.w) + (b.x + b.y + b.z + b.w) +
            (c.x + c.y + c.z + c.w) + (d.x + d.y + d.z + d.w);
}

void block_sums16_f32(const void *x_f32, void *sx16, int n16, cudaStream_t stream)
{
    if (!x_f32 || !sx16 || n16 <= 0) return;
    const int threads = 256;
    const unsigned bx = (unsigned)((n16 + threads - 1) / threads);
    block_sums16_kernel<<<bx, threads, 0, stream>>>((const float *)x_f32, (float *)sx16, n16);
}

// --- GPU logits top-k preselection ---

__device__ __forceinline__ uint32_t logit_key(float f)
{
    const uint32_t b = __float_as_uint(f);
    return b ^ (0x80000000u | (0u - (b >> 31)));
}

__global__ void logits_max_kernel(const float *x, int n, unsigned *gmax)
{
    __shared__ unsigned smax[32];
    unsigned lm = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        lm = max(lm, logit_key(x[i]));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        lm = max(lm, __shfl_down_sync(0xffffffffu, lm, o));
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (lane == 0) { smax[wid] = lm; }
    __syncthreads();
    if (threadIdx.x < 32) {
        unsigned v = (threadIdx.x < nwarps) ? smax[threadIdx.x] : 0u;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v = max(v, __shfl_down_sync(0xffffffffu, v, o));
        }
        if (threadIdx.x == 0 && v != 0u) { atomicMax(gmax, v); }
    }
}

__global__ void logits_collect_kernel(const float *x, int n, float thresh,
                                      unsigned *counter, int *idx, float *val, int cap)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (x[i] > thresh) {
            const unsigned p = atomicAdd(counter, 1u);
            if ((int)p < cap) {
                idx[p] = i;
                val[p] = x[i];
            }
        }
    }
}

__device__ __forceinline__ float logit_key_to_float(uint32_t k)
{
    const uint32_t b = k ^ ((k & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu);
    return __uint_as_float(b);
}

__host__ __device__ __forceinline__ float logit_key_to_float_host(uint32_t k)
{
    const uint32_t b = k ^ ((k & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu);
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

// Single-pass variant: reads the global max computed by logits_max_kernel
// directly on device, derives the threshold, and collects candidates packed
// as {idx, val_bits} pairs — no host round-trip between max and collect.
__global__ void logits_collect_packed_kernel(const float *x, int n, const unsigned *gmax,
                                             float window, unsigned *counter,
                                             uint2 *out, int cap)
{
    const float mx = logit_key_to_float(*gmax);   // gmax holds the order-preserving key, not raw bits
    const float thresh = mx - window;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (x[i] > thresh) {
            const unsigned p = atomicAdd(counter, 1u);
            if ((int)p < cap) {
                out[p] = make_uint2((unsigned)i, __float_as_uint(x[i]));
            }
        }
    }
}

int topk_logits_f32(const void *logits_gpu, int vocab, int need, int cap,
                    int *idx_out, float *val_out, cudaStream_t stream)
{
    if (!logits_gpu || !idx_out || !val_out || vocab <= 0 || need <= 0 || cap <= 0) { return -1; }
    static unsigned *d_gmax = nullptr;
    static unsigned *d_counter = nullptr;
    static int *d_idx = nullptr;
    static float *d_val = nullptr;
    static uint2 *d_packed = nullptr;
    static int d_cap = 0;
    if (d_gmax == nullptr) {
        CHECK_CUDA(cudaMalloc(&d_gmax, sizeof(unsigned)));
        CHECK_CUDA(cudaMalloc(&d_counter, sizeof(unsigned)));
    }
    if (d_cap < cap) {
        if (d_idx) { cudaFree(d_idx); cudaFree(d_val); cudaFree(d_packed); }
        CHECK_CUDA(cudaMalloc(&d_idx, cap * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_val, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_packed, cap * sizeof(uint2)));
        d_cap = cap;
    }
    static std::vector<uint2> h_packed;
    if ((int)h_packed.size() < cap) { h_packed.resize(cap); }
    const float *x = static_cast<const float *>(logits_gpu);
    const int threads = 256;
    const unsigned bx = (unsigned)((vocab + threads - 1) / threads);
    const unsigned gbx = bx < 4096u ? bx : 4096u;

    // Fast path: max kernel -> packed collect kernel back-to-back on the
    // same stream (the collect kernel reads gmax on device), then ONE sync
    // covering both D2H copies. Saves ~2-3 stream syncs per token vs the
    // iterative threshold loop below.
    CHECK_CUDA(cudaMemsetAsync(d_gmax, 0, sizeof(unsigned), stream));
    CHECK_CUDA(cudaMemsetAsync(d_counter, 0, sizeof(unsigned), stream));
    logits_max_kernel<<<gbx, threads, 0, stream>>>(x, vocab, d_gmax);
    logits_collect_packed_kernel<<<gbx, threads, 0, stream>>>(
        x, vocab, d_gmax, 20.0f, d_counter, d_packed, cap);
    unsigned h_count = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h_count, d_counter, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_packed.data(), d_packed, cap * sizeof(uint2), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const int fast_count = (int)h_count;
    if (fast_count >= need && fast_count <= cap) {
        for (int i = 0; i < fast_count; ++i) {
            idx_out[i] = (int)h_packed[i].x;
            float fv;
            std::memcpy(&fv, &h_packed[i].y, sizeof(fv));
            val_out[i] = fv;
        }
        return fast_count;
    }

    // Fallback: window 25 didn't land in [need, cap] — recover max on host
    // and run the adaptive iterative path.
    unsigned h_key = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h_key, d_gmax, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const float max_val = logit_key_to_float_host(h_key);

    // Collect candidates above max - window; widen until >= need (or shrink
    // if the bucket overflows `cap`).
    float window = 25.0f;
    int count = 0;
    for (int iter = 0; iter < 12; ++iter) {
        const float thresh = max_val - window;
        CHECK_CUDA(cudaMemsetAsync(d_counter, 0, sizeof(unsigned), stream));
        logits_collect_kernel<<<gbx, threads, 0, stream>>>(x, vocab, thresh, d_counter, d_idx, d_val, cap);
        unsigned h_count2 = 0;
        CHECK_CUDA(cudaMemcpyAsync(&h_count2, d_counter, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        count = (int)h_count2;
        if (count >= need && count <= cap) { break; }
        // Gentler adjustment avoids oscillating across the [need, cap] band.
        window *= (count < need) ? 1.6f : 0.6f;
        if (window > 1e6f || window < 1e-3f) { break; }
    }
    if (count < need || count > cap) { return -1; }

    const int ncopy = count;
    CHECK_CUDA(cudaMemcpyAsync(idx_out, d_idx, ncopy * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(val_out, d_val, ncopy * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return ncopy;
}

// --- Cooperative top-k selection over collected candidates ---
//
// The candidate set from logits_collect_packed_kernel is unordered and can
// hold thousands of entries; the CPU sampler only needs a superset of the
// penalized top_k (~40). This kernel binary-searches a logit threshold on
// device (grid.sync between passes) and compacts the above-threshold set —
// i.e. the exact top-N raw logits — into `out`. All threads track lo/hi/mid
// identically so no threshold state needs to live in global memory.

__global__ void topk_select_coop_kernel(const uint2 *pk, const unsigned *count_p,
                                        const unsigned *gmax,
                                        unsigned *cnts, float *g_state,
                                        uint2 *out, int max_out, int ncap)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    // count_p counts EVERY element above threshold, including ones beyond
    // the packed buffer's `cap` — clamp reads, and flag overflow so the host
    // falls back (the packed subset would be arbitrary, not the true top-N).
    const unsigned raw_n = *count_p;
    const bool overflow = raw_n > (unsigned)ncap;
    const int n = (int)min(raw_n, (unsigned)ncap);
    const float max_val = logit_key_to_float(*gmax);
    float lo = max_val - 100.0f, hi = max_val;
    float mid = 0.5f * (lo + hi);
    float sel_mid = mid;
    bool done = false;
    const unsigned TARGET_LO = 640, TARGET_HI = 1920;  // accept window

    __shared__ unsigned smem[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    // cnts[0]/cnts[1] ping-pong per iteration: each counter is only written
    // during its count phase and only reset after a second grid.sync, so the
    // threshold `total` every thread reads is race-free and identical —
    // divergent lo/hi/mid would deadlock grid.sync().
    for (int it = 0; it < 22 && !done && n > 0; ++it) {
        unsigned *cnt = cnts + (it & 1);
        unsigned c = 0;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            if (__uint_as_float(pk[i].y) > mid) { ++c; }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) { c += __shfl_down_sync(0xffffffffu, c, o); }
        if (lane == 0) { smem[wid] = c; }
        __syncthreads();
        if (wid == 0) {
            c = (lane < (blockDim.x >> 5)) ? smem[lane] : 0u;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) { c += __shfl_down_sync(0xffffffffu, c, o); }
            if (lane == 0) { atomicAdd(cnt, c); }
        }
        __syncthreads();   // smem reuse guard for next iteration
        grid.sync();       // counts final; *cnt stable until the reset below
        const unsigned total = *cnt;
        if (total >= TARGET_LO && total <= TARGET_HI) { done = true; sel_mid = mid; }
        else if (total > TARGET_HI) { lo = mid; } else { hi = mid; }
        mid = 0.5f * (lo + hi);
        grid.sync();       // all threads finished reading *cnt
        if (blockIdx.x == 0 && threadIdx.x == 0) { *cnt = 0; }  // reuse at it+2
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        g_state[0] = (done && !overflow) ? 1.0f : 0.0f;
        cnts[2] = 0;  // compaction slot counter
    }
    grid.sync();

    if (done) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            if (__uint_as_float(pk[i].y) > sel_mid) {
                const unsigned p = atomicAdd(&cnts[2], 1u);
                if ((int)p < max_out) { out[p] = pk[i]; }
            }
        }
    }
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        g_state[1] = (float)cnts[2];
    }
}

/// One-shot GPU candidate selection: max -> threshold collect -> cooperative
/// top-N compaction. Returns the number of candidates written to idx_out /
/// val_out (the exact top-N raw logits, unordered), or -1 on failure / when
/// the cooperative path is unavailable or the count can't be bounded.
int gpu_topk_raw_candidates(const void *logits_gpu, int vocab, int cap, int max_out,
                            int *idx_out, float *val_out, cudaStream_t stream)
{
    if (!logits_gpu || !idx_out || !val_out || vocab <= 0 || cap <= 0 || max_out <= 0) { return -1; }
    static unsigned *d_gmax = nullptr;
    static unsigned *d_ccount = nullptr;   // collect count (n candidates)
    static unsigned *d_cnts = nullptr;     // coop counters: [0],[1] ping-pong, [2] compaction
    static uint2 *d_packed = nullptr;
    static uint2 *d_sel = nullptr;
    static float *d_state = nullptr;
    static int d_cap = 0, d_maxout = 0;
    if (d_gmax == nullptr) {
        CHECK_CUDA(cudaMalloc(&d_gmax, sizeof(unsigned)));
        CHECK_CUDA(cudaMalloc(&d_ccount, sizeof(unsigned)));
        CHECK_CUDA(cudaMalloc(&d_cnts, 4 * sizeof(unsigned)));
        CHECK_CUDA(cudaMalloc(&d_state, 2 * sizeof(float)));
    }
    if (d_cap < cap) {
        if (d_packed) { cudaFree(d_packed); }
        CHECK_CUDA(cudaMalloc(&d_packed, cap * sizeof(uint2)));
        d_cap = cap;
    }
    if (d_maxout < max_out) {
        if (d_sel) { cudaFree(d_sel); }
        CHECK_CUDA(cudaMalloc(&d_sel, max_out * sizeof(uint2)));
        d_maxout = max_out;
    }
    static std::vector<uint2> h_sel;
    static float h_state[2];
    if ((int)h_sel.size() < max_out) { h_sel.resize(max_out); }

    // Cooperative-launch capability (queried once).
    static int coop_grid = -2;
    if (coop_grid == -2) {
        int dev = 0, numBlocksPerSm = 0, smCount = 0;
        cudaGetDevice(&dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                    (const void *)topk_select_coop_kernel, 256, 0);
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev);
        coop_grid = (numBlocksPerSm > 0 && smCount > 0) ? numBlocksPerSm * smCount : 0;
        if (coop_grid > 64) { coop_grid = 64; }
        if (coop_grid < 1) { coop_grid = -1; }
    }
    if (coop_grid < 0) { return -1; }

    const float *x = static_cast<const float *>(logits_gpu);
    const int threads = 256;
    const unsigned bx = (unsigned)((vocab + threads - 1) / threads);
    const unsigned gbx = bx < 4096u ? bx : 4096u;

    CHECK_CUDA(cudaMemsetAsync(d_gmax, 0, sizeof(unsigned), stream));
    CHECK_CUDA(cudaMemsetAsync(d_ccount, 0, sizeof(unsigned), stream));
    CHECK_CUDA(cudaMemsetAsync(d_cnts, 0, 4 * sizeof(unsigned), stream));
    logits_max_kernel<<<gbx, threads, 0, stream>>>(x, vocab, d_gmax);
    logits_collect_packed_kernel<<<gbx, threads, 0, stream>>>(
        x, vocab, d_gmax, 20.0f, d_ccount, d_packed, cap);

    void *args[] = { &d_packed, &d_ccount, &d_gmax, &d_cnts, &d_state, &d_sel, &max_out, &cap };
    cudaError_t lerr = cudaLaunchCooperativeKernel(
        (const void *)topk_select_coop_kernel, dim3((unsigned)coop_grid), dim3(256),
        args, 0, stream);
    if (lerr != cudaSuccess) { cudaGetLastError(); return -1; }

    CHECK_CUDA(cudaMemcpyAsync(h_state, d_state, 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_sel.data(), d_sel, max_out * sizeof(uint2), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (h_state[0] != 1.0f) { return -1; }
    const int ncopy = (int)h_state[1];
    if (ncopy <= 0 || ncopy > max_out) { return -1; }
    for (int i = 0; i < ncopy; ++i) {
        idx_out[i] = (int)h_sel[i].x;
        float fv;
        std::memcpy(&fv, &h_sel[i].y, sizeof(fv));
        val_out[i] = fv;
    }
    return ncopy;
}

void gemv_quant_f16(const void *x, const void *w, int qtype, size_t in, size_t out, void *y_f32,
                    cudaStream_t stream)
{
    if (!x || !w || !y_f32 || in == 0 || out == 0 || (in % 32) != 0) return;
    const int bpr = (int)(in / 32);
    const int threads = 256;
    const unsigned bx = (unsigned)((out + threads - 1) / threads);
    const size_t col_blocks = bx;
    int ksplit = (int)((384 + col_blocks - 1) / col_blocks);
    if (ksplit < 1) ksplit = 1;
    if (ksplit > bpr) ksplit = bpr;
    dim3 grid(bx, (unsigned)ksplit);

    if (qtype == 3) {
        gemv_quant_kernel<3><<<grid, threads, 0, stream>>>((const __half *)x, (const uint8_t *)w, (float *)y_f32,
                                                        (int)in, (int)out, bpr, ksplit);
    } else if (qtype == 4) {
        gemv_quant_kernel<4><<<grid, threads, 0, stream>>>((const __half *)x, (const uint8_t *)w, (float *)y_f32,
                                                        (int)in, (int)out, bpr, ksplit);
    } else if (qtype == 5) {
        gemv_quant_kernel<5><<<grid, threads, 0, stream>>>((const __half *)x, (const uint8_t *)w, (float *)y_f32,
                                                        (int)in, (int)out, bpr, ksplit);
    }
}

// --- RoPE ---

template <typename T>
__global__ void rope_kernel(T *data, int head_dim, int n_heads, float base, int pos_offset)
{
    // Rotate-half convention (matches CPU apply_rope_inplace):
    // pair i in [0, head_dim/2) rotates elements (i, i + head_dim/2).
    const int half_dim = head_dim / 2;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int i = threadIdx.x;

    if (i < half_dim) {
        float pos = (float)(seq_idx + pos_offset);
        float freq = 1.0f / powf(base, (float)(2 * i) / (float)head_dim);
        float theta = pos * freq;
        float cos_v = cosf(theta);
        float sin_v = sinf(theta);

        int head_base = (seq_idx * n_heads * head_dim) + (head_idx * head_dim);
        int offset0 = head_base + i;
        int offset1 = head_base + i + half_dim;
        float x0 = to_float(data[offset0]);
        float x1 = to_float(data[offset1]);

        data[offset0] = (T)(x0 * cos_v - x1 * sin_v);
        data[offset1] = (T)(x0 * sin_v + x1 * cos_v);
    }
}

template <typename T>
__global__ void rope_devpos_kernel(T *data, int head_dim, int n_heads, float base, const int *pos_dev)
{
    const int half_dim = head_dim / 2;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int i = threadIdx.x;
    const int pos_offset = *pos_dev;

    if (i < half_dim) {
        float pos = (float)(seq_idx + pos_offset);
        float freq = 1.0f / powf(base, (float)(2 * i) / (float)head_dim);
        float theta = pos * freq;
        float cos_v = cosf(theta);
        float sin_v = sinf(theta);

        int head_base = (seq_idx * n_heads * head_dim) + (head_idx * head_dim);
        int offset0 = head_base + i;
        int offset1 = head_base + i + half_dim;
        float x0 = to_float(data[offset0]);
        float x1 = to_float(data[offset1]);

        data[offset0] = (T)(x0 * cos_v - x1 * sin_v);
        data[offset1] = (T)(x0 * sin_v + x1 * cos_v);
    }
}

void rope_devpos(void *data, size_t head_dim, size_t seq_len, size_t n_heads, float base,
                 const int *pos_dev, DType dtype, cudaStream_t stream)
{
    if (!data || !pos_dev || head_dim == 0 || seq_len == 0 || n_heads == 0) return;

    dim3 grid((unsigned int)n_heads, (unsigned int)seq_len);
    int threads = (int)((head_dim / 2 + 31) / 32 * 32);
    if (threads > 1024) threads = 1024;

    if (dtype == DType::Float16) {
        rope_devpos_kernel<<<grid, threads, 0, stream>>>((__half *)data, (int)head_dim, (int)n_heads, base, pos_dev);
    } else if (dtype == DType::BFloat16) {
        rope_devpos_kernel<<<grid, threads, 0, stream>>>((__nv_bfloat16 *)data, (int)head_dim, (int)n_heads, base, pos_dev);
    } else {
        rope_devpos_kernel<<<grid, threads, 0, stream>>>((float *)data, (int)head_dim, (int)n_heads, base, pos_dev);
    }
}

template <typename T>
__global__ void rope_bias_devpos_kernel(T *data, int head_dim, int n_heads, float base,
                                        const int *pos_dev, const T *bias, float scale)
{
    const int half_dim = head_dim / 2;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int i = threadIdx.x;
    const int pos_offset = *pos_dev;

    if (i < half_dim) {
        // Bias is per-column within a row: index repeats for every seq position.
        const int b0 = head_idx * head_dim + i;
        const int b1 = b0 + half_dim;
        const float fb0 = bias ? to_float(bias[b0]) : 0.0f;
        const float fb1 = bias ? to_float(bias[b1]) : 0.0f;

        float pos = (float)(seq_idx + pos_offset);
        float freq = 1.0f / powf(base, (float)(2 * i) / (float)head_dim);
        float theta = pos * freq;
        float cos_v = cosf(theta);
        float sin_v = sinf(theta);

        int head_base = (seq_idx * n_heads * head_dim) + (head_idx * head_dim);
        int offset0 = head_base + i;
        int offset1 = head_base + i + half_dim;
        float x0 = to_float(data[offset0]) + fb0;
        float x1 = to_float(data[offset1]) + fb1;

        float r0 = (x0 * cos_v - x1 * sin_v) * scale;
        float r1 = (x0 * sin_v + x1 * cos_v) * scale;
        data[offset0] = (T)r0;
        data[offset1] = (T)r1;
    }
}

void rope_bias_devpos(void *data, size_t head_dim, size_t seq_len, size_t n_heads, float base,
                      const int *pos_dev, const void *bias, float scale, DType dtype,
                      cudaStream_t stream)
{
    if (!data || !pos_dev || head_dim == 0 || seq_len == 0 || n_heads == 0) return;

    dim3 grid((unsigned int)n_heads, (unsigned int)seq_len);
    int threads = (int)((head_dim / 2 + 31) / 32 * 32);
    if (threads > 1024) threads = 1024;

    if (dtype == DType::Float16) {
        rope_bias_devpos_kernel<<<grid, threads, 0, stream>>>(
            (__half *)data, (int)head_dim, (int)n_heads, base, pos_dev,
            (const __half *)bias, scale);
    } else if (dtype == DType::BFloat16) {
        rope_bias_devpos_kernel<<<grid, threads, 0, stream>>>(
            (__nv_bfloat16 *)data, (int)head_dim, (int)n_heads, base, pos_dev,
            (const __nv_bfloat16 *)bias, scale);
    } else {
        rope_bias_devpos_kernel<<<grid, threads, 0, stream>>>(
            (float *)data, (int)head_dim, (int)n_heads, base, pos_dev,
            (const float *)bias, scale);
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
__global__ void add_row_bias_kernel(T *out, const T *bias, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        out[i] = (T)(to_float(out[i]) + to_float(bias[i % cols]));
    }
}

void add_row_bias(void *out, const void *bias, size_t rows, size_t cols, DType dtype, cudaStream_t stream)
{
    if (!out || !bias || rows == 0 || cols == 0) return;
    int threads = 256;
    int blocks = (int)((rows * cols + threads - 1) / threads);

    if (dtype == DType::Float32) {
        add_row_bias_kernel<<<blocks, threads, 0, stream>>>((float *)out, (const float *)bias, (int)rows, (int)cols);
    } else if (dtype == DType::Float16) {
        add_row_bias_kernel<<<blocks, threads, 0, stream>>>((__half *)out, (const __half *)bias, (int)rows, (int)cols);
    } else if (dtype == DType::BFloat16) {
        add_row_bias_kernel<<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)bias, (int)rows, (int)cols);
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

    // Persistent device staging buffer for shape/strides: avoids cudaMalloc +
    // stream sync + cudaFree on every call (this runs dozens of times per token).
    constexpr int kMaxDims = 16;
    static size_t *d_meta = nullptr;
    if (d_meta == nullptr) {
        CHECK_CUDA(cudaMalloc(&d_meta, 2 * kMaxDims * sizeof(size_t)));
    }
    if (dims > kMaxDims) return;
    CHECK_CUDA(cudaMemcpyAsync(d_meta, shape, dims * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_meta + dims, src_strides, dims * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    const size_t *d_shape = d_meta;
    const size_t *d_strides = d_meta + dims;

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    if (dtype == DType::Float32) {
        copy_layout_kernel<<<blocks, threads, 0, stream>>>((float *)dst, (const float *)src, n, (int)dims, d_shape, d_strides);
    } else if (dtype == DType::Float16) {
        copy_layout_kernel<<<blocks, threads, 0, stream>>>((__half *)dst, (const __half *)src, n, (int)dims, d_shape, d_strides);
    } else if (dtype == DType::BFloat16) {
        copy_layout_kernel<<<blocks, threads, 0, stream>>>((__nv_bfloat16 *)dst, (const __nv_bfloat16 *)src, n, (int)dims, d_shape, d_strides);
    }
}

namespace {

// Fused single-token GQA decode attention (FP16): replaces repeat_kv +
// transpose/contiguous copies + 2x batched_matmul + softmax + output
// transpose with ONE kernel per layer.
//   q:     [n_heads*head_dim] FP16, already RoPE'd and pre-scaled by 1/sqrt(head_dim)
//   K,V:   [max_seq, kv_dim] FP16 row-major KV cache (kv_dim = n_kv*head_dim)
//   out:   [n_heads*head_dim] FP16
// kv head for query head h: h / (n_heads / n_kv)  (matches repeat_kv_kernel).
// pos = number of valid cache rows (effective_len). Requires pos <= 4096.
// Shared-memory layout for the fused decode attention: w[pos] scores at the
// front, then the FP32 query copy and reduction scratch at a FIXED offset so
// the same kernel works whether pos comes from a launch arg or device memory.
__device__ __forceinline__ void gqa_decode_attn_f16_impl(const __half *q, const __half *K,
                                                       const __half *V, __half *out, int pos,
                                                       int n_heads, int n_kv, int head_dim,
                                                       int kv_dim, int qf_off, float *sm)
{
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kvh = head / (n_heads / n_kv);
    const int hd = head_dim;

    float *w = sm;                      // pos scores/weights
    float *qf = sm + qf_off;            // head_dim query floats
    float *red = qf + hd;               // reduction scratch (24 floats)

    if (tid < hd) { qf[tid] = __half2float(q[head * hd + tid]); }
    __syncthreads();

    // Phase 1: scores s[p] = <q_head, K[p, kvh]>
    const int kvoff = kvh * hd;
    for (int p = tid; p < pos; p += nthreads) {
        const __half *krow = K + (size_t)p * kv_dim + kvoff;
        float s = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < hd; ++d) {
            s = fmaf(qf[d], __half2float(krow[d]), s);
        }
        w[p] = s;
    }
    __syncthreads();

    // Phase 2: softmax — max reduce (shared-mem, no float atomicMax)
    float lmax = -CUDART_INF_F;
    for (int p = tid; p < pos; p += nthreads) { lmax = fmaxf(lmax, w[p]); }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { lmax = fmaxf(lmax, __shfl_down_sync(0xffffffffu, lmax, o)); }
    const int wid = tid >> 5;
    const int nwarp = (nthreads + 31) >> 5;
    if ((tid & 31) == 0) { red[1 + wid] = lmax; }
    __syncthreads();
    if (tid == 0) {
        float mx = -CUDART_INF_F;
        for (int i = 0; i < nwarp; ++i) { mx = fmaxf(mx, red[1 + i]); }
        red[0] = mx;
    }
    __syncthreads();
    const float mx = red[0];

    // exp + sum reduce
    float lsum = 0.0f;
    for (int p = tid; p < pos; p += nthreads) {
        const float e = expf(w[p] - mx);
        w[p] = e;
        lsum += e;
    }
    // block reduce lsum
    for (int o = 16; o > 0; o >>= 1) { lsum += __shfl_down_sync(0xffffffffu, lsum, o); }
    if ((tid & 31) == 0) { red[9 + wid] = lsum; }
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < nwarp; ++i) { s += red[9 + i]; }
        red[0] = s;
    }
    __syncthreads();
    const float inv_sum = 1.0f / red[0];
    for (int p = tid; p < pos; p += nthreads) { w[p] *= inv_sum; }
    __syncthreads();

    // Phase 3: out[d] = sum_p w[p] * V[p, kvh, d]
    if (tid < hd) {
        float acc = 0.0f;
        for (int p = 0; p < pos; ++p) {
            acc = fmaf(w[p], __half2float(V[(size_t)p * kv_dim + kvoff + tid]), acc);
        }
        out[head * hd + tid] = __float2half(acc);
    }
}

#define GQA_DEC_ATTN_MAX_POS 4096

__global__ void gqa_decode_attn_f16_kernel(const __half *q, const __half *K, const __half *V,
                                           __half *out, int pos, int n_heads, int n_kv,
                                           int head_dim, int kv_dim)
{
    extern __shared__ float sm[];
    gqa_decode_attn_f16_impl(q, K, V, out, pos, n_heads, n_kv, head_dim, kv_dim, pos, sm);
}

__global__ void gqa_decode_attn_f16_devpos_kernel(const __half *q, const __half *K, const __half *V,
                                                  __half *out, const int *pos_dev, int n_heads,
                                                  int n_kv, int head_dim, int kv_dim)
{
    extern __shared__ float sm[];
    gqa_decode_attn_f16_impl(q, K, V, out, (*pos_dev) + 1, n_heads, n_kv, head_dim, kv_dim,
                             GQA_DEC_ATTN_MAX_POS, sm);
}

__global__ void kv_write_devpos_kernel(char *dst_base, const char *src, size_t nbytes,
                                       const int *pos_dev, size_t row_stride)
{
    const size_t pos = (size_t)(*pos_dev);
    char *dst = dst_base + pos * row_stride;
    const int nvec = (int)(nbytes >> 4);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nvec) {
        reinterpret_cast<uint4 *>(dst)[tid] = reinterpret_cast<const uint4 *>(src)[tid];
    }
    const int tail = (int)(nbytes & 15);
    if (tail != 0 && blockIdx.x == 0 && threadIdx.x == blockDim.x - 1) {
        char *d = dst + nvec * 16;
        const char *s = src + nvec * 16;
        for (int i = 0; i < tail; ++i) d[i] = s[i];
    }
}

template <typename T>
__global__ void kv_write_bias_devpos_kernel(T *dst_base, const T *src, const T *bias,
                                            size_t n, const int *pos_dev, size_t row_stride)
{
    const size_t pos = (size_t)(*pos_dev);
    T *dst = dst_base + pos * row_stride;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (size_t i = (size_t)tid; i < n; i += (size_t)stride) {
        dst[i] = bias ? (T)(to_float(src[i]) + to_float(bias[i])) : src[i];
    }
}

}  // namespace

void kv_write_bias_devpos(void *dst_base, const void *src, const void *bias, size_t n_elems,
                          const int *pos_dev, size_t row_stride_elems, DType dtype,
                          cudaStream_t stream)
{
    if (!dst_base || !src || !pos_dev || n_elems == 0 || row_stride_elems == 0) return;
    const int nthreads = 256;
    const int blocks = (int)((n_elems + nthreads - 1) / nthreads);
    if (dtype == DType::Float16) {
        kv_write_bias_devpos_kernel<__half><<<blocks, nthreads, 0, stream>>>(
            (__half *)dst_base, (const __half *)src, (const __half *)bias, n_elems, pos_dev,
            row_stride_elems);
    } else if (dtype == DType::BFloat16) {
        kv_write_bias_devpos_kernel<__nv_bfloat16><<<blocks, nthreads, 0, stream>>>(
            (__nv_bfloat16 *)dst_base, (const __nv_bfloat16 *)src, (const __nv_bfloat16 *)bias,
            n_elems, pos_dev, row_stride_elems);
    } else {
        kv_write_bias_devpos_kernel<float><<<blocks, nthreads, 0, stream>>>(
            (float *)dst_base, (const float *)src, (const float *)bias, n_elems, pos_dev,
            row_stride_elems);
    }
}

void gqa_decode_attention_f16(const void *q, const void *K, const void *V, void *out,
                              int pos, int n_heads, int n_kv, int head_dim, int kv_dim,
                              cudaStream_t stream)
{
    if (!q || !K || !V || !out || pos <= 0 || pos > 4096) return;
    const int nthreads = 256;
    const size_t shmem = (size_t)(pos + head_dim + 24) * sizeof(float);
    gqa_decode_attn_f16_kernel<<<n_heads, nthreads, shmem, stream>>>(
        (const __half *)q, (const __half *)K, (const __half *)V, (__half *)out,
        pos, n_heads, n_kv, head_dim, kv_dim);
}

void gqa_decode_attention_f16_devpos(const void *q, const void *K, const void *V, void *out,
                                     const int *pos_dev, int n_heads, int n_kv, int head_dim,
                                     int kv_dim, cudaStream_t stream)
{
    if (!q || !K || !V || !out || !pos_dev || n_heads <= 0) return;
    const int nthreads = 256;
    // Fixed max-size shared memory: qf sits at offset MAX_POS; w[] uses pos floats.
    const size_t shmem = (size_t)(GQA_DEC_ATTN_MAX_POS + head_dim + 24) * sizeof(float);
    gqa_decode_attn_f16_devpos_kernel<<<n_heads, nthreads, shmem, stream>>>(
        (const __half *)q, (const __half *)K, (const __half *)V, (__half *)out,
        pos_dev, n_heads, n_kv, head_dim, kv_dim);
}

void kv_write_devpos(void *dst_base, const void *src, size_t nbytes,
                     const int *pos_dev, size_t row_stride_bytes,
                     cudaStream_t stream)
{
    if (!dst_base || !src || !pos_dev || nbytes == 0 || row_stride_bytes == 0) return;
    const int nthreads = 256;
    const int blocks = (int)((nbytes / 16 + nthreads - 1) / nthreads);
    kv_write_devpos_kernel<<<blocks, nthreads, 0, stream>>>(
        (char *)dst_base, (const char *)src, nbytes, pos_dev, row_stride_bytes);
}

}  // namespace hal
}  // namespace serius
