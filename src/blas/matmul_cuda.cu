#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "matmul_cuda.h"

#define CHECK_CUDA_ERROR(call)                                                                       \
    do {                                                                                             \
        cudaError_t err = call;                                                                      \
        if (err != cudaSuccess) {                                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" \
                      << cudaGetErrorString(err) << "\"" << std::endl;                               \
        }                                                                                            \
    } while (0)

namespace serius {
namespace core {
namespace blas {

namespace {
cublasHandle_t g_cublas_handle = nullptr;

cublasHandle_t get_cublas_handle()
{
    if (g_cublas_handle == nullptr) {
        if (cublasCreate(&g_cublas_handle) != CUBLAS_STATUS_SUCCESS) {
            g_cublas_handle = nullptr;
        }
    }
    return g_cublas_handle;
}

cudaDataType cublas_dtype(hal::DType dtype)
{
    switch (dtype) {
        case hal::DType::Float32: return CUDA_R_32F;
        case hal::DType::Float16: return CUDA_R_16F;
        case hal::DType::BFloat16: return CUDA_R_16BF;
        default: return CUDA_R_32F;
    }
}
}  // namespace

// Helper to convert to float for calculation
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
    return (float)val;  // Built-in conversion works in modern CUDA
}

template <typename T>
__global__ void matmul_kernel_simple(const T *A, const T *B, T *C, int M, int N, int K, int lda, int ldb, int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += to_float(A[row * lda + k]) * to_float(B[k * ldb + col]);
        }
        C[row * ldc + col] = (T)sum;
    }
}

template <typename T>
__global__ void batched_matmul_kernel(const T *A, const T *B, T *C, int batch_size, int M, int N, int K, int lda,
                                      int ldb, int ldc, long long stride_a, long long stride_b, long long stride_c)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < M && col < N) {
        const T *a_batch = A + batch * stride_a;
        const T *b_batch = B + batch * stride_b;
        T *c_batch = C + batch * stride_c;

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += to_float(a_batch[row * lda + k]) * to_float(b_batch[k * ldb + col]);
        }
        c_batch[row * ldc + col] = (T)sum;
    }
}

void matmul_cuda(const void *A, const void *B, void *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                 size_t ldc, hal::DType dtype, void *stream)
{
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    // cuBLAS fast path for contiguous row-major C = A @ B.
    // Row-major (M,N) = (M,K)@(K,N) maps to col-major (N,M) = (N,K)@(K,M),
    // i.e. operands are swapped with OP_N on both sides.
    const bool contiguous = (lda == K) && (ldb == N) && (ldc == N);
    static const bool use_cublas = [] {
        char buf[2];
        size_t n = 0;
        return !(getenv_s(&n, buf, sizeof(buf), "SERIUS_NO_CUBLAS") == 0 && n > 0);
    }();
    static const bool dbg = [] { size_t n = 0; return getenv_s(&n, nullptr, 0, "SERIUS_CUBLAS_DBG") == 0 && n > 0 ? 1 : 0; }();
    if (dbg && !(contiguous && use_cublas)) {
        fprintf(stderr, "[cublas] skip M=%zu N=%zu K=%zu lda=%zu ldb=%zu ldc=%zu contiguous=%d use=%d\n",
                M, N, K, lda, ldb, ldc, (int)contiguous, (int)use_cublas);
    }
    if (contiguous && use_cublas) {
        cublasHandle_t handle = get_cublas_handle();
        if (handle == nullptr && dbg) fprintf(stderr, "[cublas] handle null\n");
        if (handle != nullptr) {
            cublasSetStream(handle, cu_stream);
            const float alpha = 1.0f, beta = 0.0f;
            const cudaDataType ty = cublas_dtype(dtype);
            cublasStatus_t st = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
                                             B, ty, (int)ldb, A, ty, (int)lda, &beta, C, ty, (int)ldc,
                                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) {
                return;
            }
            static int dbg = [] { size_t n = 0; return getenv_s(&n, nullptr, 0, "SERIUS_CUBLAS_DBG") == 0 && n > 0 ? 1 : 0; }();
            if (dbg) {
                fprintf(stderr, "[cublas] GemmEx failed st=%d M=%zu N=%zu K=%zu lda=%zu ldb=%zu ldc=%zu\n",
                        (int)st, M, N, K, lda, ldb, ldc);
            }
        }
    }

    dim3 block(16, 16);
    dim3 grid((int)((N + 15) / 16), (int)((M + 15) / 16));

    if (dtype == hal::DType::Float32) {
        matmul_kernel_simple<float><<<grid, block, 0, cu_stream>>>((const float *)A, (const float *)B, (float *)C, (int)M,
                                                                   (int)N, (int)K, (int)lda, (int)ldb, (int)ldc);
    } else if (dtype == hal::DType::Float16) {
        matmul_kernel_simple<__half><<<grid, block, 0, cu_stream>>>((const __half *)A, (const __half *)B, (__half *)C,
                                                                    (int)M, (int)N, (int)K, (int)lda, (int)ldb, (int)ldc);
    } else if (dtype == hal::DType::BFloat16) {
        matmul_kernel_simple<__nv_bfloat16><<<grid, block, 0, cu_stream>>>(
            (const __nv_bfloat16 *)A, (const __nv_bfloat16 *)B, (__nv_bfloat16 *)C, (int)M, (int)N, (int)K, (int)lda,
            (int)ldb, (int)ldc);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
}

void batched_matmul_cuda(const void *A, const void *B, void *C, size_t batch_size, size_t M, size_t N, size_t K,
                         size_t lda, size_t ldb, size_t ldc, size_t stride_a, size_t stride_b, size_t stride_c,
                         hal::DType dtype, void *stream)
{
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    // cuBLAS strided-batched fast path (same row-major swap as matmul).
    const bool contiguous = (lda == K) && (ldb == N) && (ldc == N) &&
                            (stride_a == M * K) && (stride_c == M * N) &&
                            (stride_b == 0 || stride_b == K * N);
    if (contiguous && batch_size > 1) {
        cublasHandle_t handle = get_cublas_handle();
        if (handle != nullptr) {
            cublasSetStream(handle, cu_stream);
            const float alpha = 1.0f, beta = 0.0f;
            const cudaDataType ty = cublas_dtype(dtype);
            cublasStatus_t st = cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
                B, ty, (int)ldb, (long long)stride_b, A, ty, (int)lda, (long long)stride_a, &beta,
                C, ty, (int)ldc, (long long)stride_c, (int)batch_size, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) {
                return;
            }
        }
    }

    dim3 block(16, 16, 1);
    dim3 grid((int)((N + 15) / 16), (int)((M + 15) / 16), (int)batch_size);

    if (dtype == hal::DType::Float32) {
        batched_matmul_kernel<float><<<grid, block, 0, cu_stream>>>((const float *)A, (const float *)B, (float *)C,
                                                                    (int)batch_size, (int)M, (int)N, (int)K, (int)lda,
                                                                    (int)ldb, (int)ldc, (long long)stride_a,
                                                                    (long long)stride_b, (long long)stride_c);
    } else if (dtype == hal::DType::Float16) {
        batched_matmul_kernel<__half><<<grid, block, 0, cu_stream>>>(
            (const __half *)A, (const __half *)B, (__half *)C, (int)batch_size, (int)M, (int)N, (int)K, (int)lda,
            (int)ldb, (int)ldc, (long long)stride_a, (long long)stride_b, (long long)stride_c);
    } else if (dtype == hal::DType::BFloat16) {
        batched_matmul_kernel<__nv_bfloat16><<<grid, block, 0, cu_stream>>>(
            (const __nv_bfloat16 *)A, (const __nv_bfloat16 *)B, (__nv_bfloat16 *)C, (int)batch_size, (int)M, (int)N,
            (int)K, (int)lda, (int)ldb, (int)ldc, (long long)stride_a, (long long)stride_b, (long long)stride_c);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
}

void logits_gemv_mixed_cublas(const void *x, const void *w, void *out, size_t seq, size_t vocab, size_t hidden,
                              void *stream)
{
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    cublasHandle_t handle = get_cublas_handle();
    if (handle == nullptr) return;
    cublasSetStream(handle, cu_stream);

    // Row-major out[seq, vocab] = x[seq, hidden] @ w[vocab, hidden]^T.
    // In col-major terms this is C'(vocab x seq) = A'(hidden x vocab)^T * B'(hidden x seq)
    // where A' reuses the w memory with lda=hidden, B' reuses x with ldb=hidden,
    // and C' with ldc=vocab is exactly the row-major out layout.
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)vocab, (int)seq, (int)hidden, &alpha,
                                     w, CUDA_R_16F, (int)hidden, x, CUDA_R_16F, (int)hidden, &beta,
                                     out, CUDA_R_32F, (int)vocab, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cublas] logits_gemv_mixed failed st=" << (int)st << std::endl;
    }
}

}  // namespace blas
}  // namespace core
}  // namespace serius
