#pragma once

#include <cstddef>

namespace serius {
namespace blas {

void matmul_cuda(const float *A, const float *B, float *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                 size_t ldc);

void batched_matmul_cuda(const float *A, const float *B, float *C, size_t batch_size, size_t M, size_t N, size_t K,
                         size_t lda, size_t ldb, size_t ldc, size_t stride_a, size_t stride_b, size_t stride_c);

}  // namespace blas
}  // namespace serius
