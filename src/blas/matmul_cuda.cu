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
namespace blas {

__global__ void matmul_kernel_simple(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb,
                                     int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * lda + k] * B[k * ldb + col];
        }
        C[row * ldc + col] = sum;
    }
}

__global__ void batched_matmul_kernel(const float *A, const float *B, float *C, int batch_size, int M, int N, int K,
                                      int lda, int ldb, int ldc, long long stride_a, long long stride_b,
                                      long long stride_c)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < M && col < N) {
        const float *a_batch = A + batch * stride_a;
        const float *b_batch = B + batch * stride_b;
        float *c_batch = C + batch * stride_c;

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a_batch[row * lda + k] * b_batch[k * ldb + col];
        }
        c_batch[row * ldc + col] = sum;
    }
}

void matmul_cuda(const float *A, const float *B, float *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                 size_t ldc)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_kernel_simple<<<grid, block>>>(A, B, C, (int)M, (int)N, (int)K, (int)lda, (int)ldb, (int)ldc);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void batched_matmul_cuda(const float *A, const float *B, float *C, size_t batch_size, size_t M, size_t N, size_t K,
                         size_t lda, size_t ldb, size_t ldc, size_t stride_a, size_t stride_b, size_t stride_c)
{
    dim3 block(16, 16, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, batch_size);
    batched_matmul_kernel<<<grid, block>>>(A, B, C, (int)batch_size, (int)M, (int)N, (int)K, (int)lda, (int)ldb,
                                           (int)ldc, (long long)stride_a, (long long)stride_b, (long long)stride_c);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

}  // namespace blas
}  // namespace serius
