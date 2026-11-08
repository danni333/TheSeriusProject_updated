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
    if (M == 1) {
        // High-performance AVX2 vector-matrix multiplication
        // Process 32 columns at once to reuse broadcasted A[k]
        // Use OpenMP only for large matrices (e.g. lm_head with vocab_size 32k/49k)
        // to avoid thread synchronization overhead on small layer projections (N=576/1536).
        #pragma omp parallel for if (static_cast<int64_t>(N) * static_cast<int64_t>(K) >= 262144) schedule(static)
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
    // OpenMP parallelises across rows; each thread handles independent rows of C.
    #pragma omp parallel for schedule(static)
    for (int64_t ii = 0; ii < static_cast<int64_t>(M); ++ii) {
        size_t i = static_cast<size_t>(ii);
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

// Transposed-weight GEMV: y[j] = dot(W[j,:], x[:]), W is [N,K] row-major.
// Each row streams sequentially; x stays hot in cache across rows.
void matvec_t_avx2(const float *W, const float *x, float *y, size_t N, size_t K)
{
    #pragma omp parallel for schedule(static) if (N >= 1024)
    for (int64_t j = 0; j < static_cast<int64_t>(N); ++j) {
        const float *w = W + static_cast<size_t>(j) * K;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        size_t k = 0;
        for (; k + 32 <= K; k += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k + 0),  _mm256_loadu_ps(x + k + 0),  acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k + 8),  _mm256_loadu_ps(x + k + 8),  acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k + 16), _mm256_loadu_ps(x + k + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k + 24), _mm256_loadu_ps(x + k + 24), acc3);
        }
        alignas(32) float buf[32];
        _mm256_store_ps(buf + 0,  acc0);
        _mm256_store_ps(buf + 8,  acc1);
        _mm256_store_ps(buf + 16, acc2);
        _mm256_store_ps(buf + 24, acc3);
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) { sum += buf[i]; }
        for (; k < K; ++k) { sum += w[k] * x[k]; }
        y[static_cast<size_t>(j)] = sum;
    }
}

}  // namespace blas
}  // namespace core
}  // namespace serius
