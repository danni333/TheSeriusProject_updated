#include "blas.h"

#if defined(_MSC_VER)
#include <immintrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cstdint>

namespace serius {
namespace core {
namespace blas {

// AVX2 implementation of Matrix Multiplication
// M = rows of A, C
// N = cols of B, C
// K = cols of A, rows of B

// Microkernel for 8x8 block
static inline void microkernel_8x8(const float *A, const float *B, float *C, size_t K, size_t lda, size_t ldb,
                                   size_t ldc)
{
    __m256 c[8];
    for (int i = 0; i < 8; ++i) {
        c[i] = _mm256_setzero_ps();
    }

    for (size_t k = 0; k < K; ++k) {
        __m256 b_vec = _mm256_loadu_ps(B + k * ldb);

        for (int i = 0; i < 8; ++i) {
            __m256 a_vec = _mm256_broadcast_ss(A + i * lda + k);
            c[i] = _mm256_fmadd_ps(a_vec, b_vec, c[i]);
        }
    }

    for (int i = 0; i < 8; ++i) {
        __m256 c_old = _mm256_loadu_ps(C + i * ldc);
        __m256 c_new = _mm256_add_ps(c_old, c[i]);
        _mm256_storeu_ps(C + i * ldc, c_new);
    }
}

void matmul_avx2(const float *A, const float *B, float *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                 size_t ldc)
{
    printf("DEBUG: Entering matmul_avx2 (M=%zu, N=%zu, K=%zu)\n", M, N, K);

    if (M == 1) {
        // High-performance AVX2 vector-matrix multiplication
        // Process 32 columns at once to reuse broadcasted A[k]
        #pragma omp parallel for
        for (int64_t j = 0; j < static_cast<int64_t>(N); j += 32) {
            if (j + 32 <= static_cast<int64_t>(N)) {
                __m256 sum0 = _mm256_setzero_ps();
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m256 sum3 = _mm256_setzero_ps();

                for (size_t k = 0; k < K; ++k) {
                    __m256 a_val = _mm256_set1_ps(A[k]);
                    sum0 = _mm256_fmadd_ps(a_val, _mm256_loadu_ps(B + k * ldb + j + 0), sum0);
                    sum1 = _mm256_fmadd_ps(a_val, _mm256_loadu_ps(B + k * ldb + j + 8), sum1);
                    sum2 = _mm256_fmadd_ps(a_val, _mm256_loadu_ps(B + k * ldb + j + 16), sum2);
                    sum3 = _mm256_fmadd_ps(a_val, _mm256_loadu_ps(B + k * ldb + j + 24), sum3);
                }
                _mm256_storeu_ps(C + j + 0, sum0);
                _mm256_storeu_ps(C + j + 8, sum1);
                _mm256_storeu_ps(C + j + 16, sum2);
                _mm256_storeu_ps(C + j + 24, sum3);
            } else {
                // Fallback for remaining columns (per 8)
                for (int64_t jj = j; jj < static_cast<int64_t>(N); jj += 8) {
                    __m256 sum_vec = _mm256_setzero_ps();
                    if (jj + 8 <= static_cast<int64_t>(N)) {
                        for (size_t k = 0; k < K; ++k) {
                            sum_vec = _mm256_fmadd_ps(_mm256_set1_ps(A[k]), _mm256_loadu_ps(B + k * ldb + jj), sum_vec);
                        }
                        _mm256_storeu_ps(C + jj, sum_vec);
                    } else {
                        // Scalar cleanup for individual columns
                        for (size_t jjj = static_cast<size_t>(jj); jjj < N; ++jjj) {
                            float s = 0.0f;
                            for (size_t k = 0; k < K; ++k) {
                                s += A[k] * B[k * ldb + jjj];
                            }
                            C[jjj] = s;
                        }
                    }
                }
            }
        }
        return;
    }

    // General AVX2 matrix multiplication - process 8 columns at a time per row.
    // Each row of A is multiplied against all columns of B, fully accumulating over K.
    for (size_t i = 0; i < M; ++i) {
        size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t k = 0; k < K; ++k) {
                __m256 a_val = _mm256_set1_ps(A[i * lda + k]);
                __m256 b_vec = _mm256_loadu_ps(B + k * ldb + j);
                acc = _mm256_fmadd_ps(a_val, b_vec, acc);
            }
            _mm256_storeu_ps(C + i * ldc + j, acc);
        }
        // Scalar cleanup for remaining columns
        for (; j < N; ++j) {
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
