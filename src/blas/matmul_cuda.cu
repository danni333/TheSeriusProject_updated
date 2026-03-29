#include <cuda_bf16.h>
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
    dim3 block(16, 16);
    dim3 grid((int)((N + 15) / 16), (int)((M + 15) / 16));

    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

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
    dim3 block(16, 16, 1);
    dim3 grid((int)((N + 15) / 16), (int)((M + 15) / 16), (int)batch_size);

    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

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

}  // namespace blas
}  // namespace core
}  // namespace serius
