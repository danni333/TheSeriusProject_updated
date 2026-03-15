#include "blas.h"
#include <cstdint>

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
    if (M == 1) {
        // High-performance path for token generation (M=1)
        // Restructured to use tiled row-major accumulation (weighted sum of rows)
        // to maximize cache line utilization on the weight matrix B.
        #pragma omp parallel for
        for (int64_t j_start = 0; j_start < static_cast<int64_t>(N); j_start += 64) {
            int64_t j_end = (j_start + 64 < static_cast<int64_t>(N)) ? (j_start + 64) : static_cast<int64_t>(N);
            
            // Local zeroing of this chunk of C
            for (int64_t j = j_start; j < j_end; ++j) {
                C[j] = 0.0f;
            }

            for (size_t k = 0; k < K; ++k) {
                float a_val = A[k];
                const float *b_row = B + k * ldb;
                for (int64_t j = j_start; j < j_end; ++j) {
                    C[j] += a_val * b_row[j];
                }
            }
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(M); ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[i * lda + k] * B[k * ldb + j];
                }
                C[i * ldc + j] = sum;
            }
        }
    }
}

}  // namespace blas
}  // namespace core
}  // namespace serius
