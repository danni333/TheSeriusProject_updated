#include "blas.h"

namespace serius {
namespace core {
namespace blas {

// Naive scalar matrix multiplication
// C = A * B
// A is M x K
// B is K x N
// C is M x N
void matmul_scalar(const float *A, const float *B, float *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                   size_t ldc)
{
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

}  // namespace blas
}  // namespace core
}  // namespace serius
