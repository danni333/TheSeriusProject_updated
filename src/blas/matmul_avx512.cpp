#include "blas.h"

#if defined(_MSC_VER)
#include <immintrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#endif

#include <algorithm>

namespace serius {
namespace core {
namespace blas {

// AVX-512 implementation of Matrix Multiplication
// 16 floats per vector register. We can do 16x16 microkernels efficiently on AVX-512
// since there are 32 zmm registers!

static inline void microkernel_16x16(const float *A, const float *B, float *C, size_t K, size_t lda, size_t ldb,
                                     size_t ldc)
{
#if defined(__AVX512F__) || defined(_MSC_VER)
    __m512 c[16];
    for (int i = 0; i < 16; ++i) {
        c[i] = _mm512_setzero_ps();
    }

    for (size_t k = 0; k < K; ++k) {
        __m512 b_vec = _mm512_loadu_ps(B + k * ldb);

        for (int i = 0; i < 16; ++i) {
            __m512 a_vec = _mm512_set1_ps(A[i * lda + k]);
            c[i] = _mm512_fmadd_ps(a_vec, b_vec, c[i]);
        }
    }

    for (int i = 0; i < 16; ++i) {
        __m512 c_old = _mm512_loadu_ps(C + i * ldc);
        __m512 c_new = _mm512_add_ps(c_old, c[i]);
        _mm512_storeu_ps(C + i * ldc, c_new);
    }
#else
    // Fallback if compiled without AVX512 flags but called anyway (shouldn't happen with correct CMake)
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += sum;
        }
    }
#endif
}

void matmul_avx512(const float *A, const float *B, float *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                   size_t ldc)
{
    if (M == 1) {
#if defined(__AVX512F__) || defined(_MSC_VER)
        // High-performance AVX-512 vector-matrix multiplication
        // Process 64 columns at once to reuse broadcasted A[k]
        // AVX-512 has 16 floats per vector (512 bits)
        #pragma omp parallel for
        for (int64_t j = 0; j < static_cast<int64_t>(N); j += 64) {
            if (j + 64 <= static_cast<int64_t>(N)) {
                __m512 sum0 = _mm512_setzero_ps();
                __m512 sum1 = _mm512_setzero_ps();
                __m512 sum2 = _mm512_setzero_ps();
                __m512 sum3 = _mm512_setzero_ps();

                for (size_t k = 0; k < K; ++k) {
                    __m512 a_val = _mm512_set1_ps(A[k]);
                    sum0 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(B + k * ldb + j + 0), sum0);
                    sum1 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(B + k * ldb + j + 16), sum1);
                    sum2 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(B + k * ldb + j + 32), sum2);
                    sum3 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(B + k * ldb + j + 48), sum3);
                }
                _mm512_storeu_ps(C + j + 0, sum0);
                _mm512_storeu_ps(C + j + 16, sum1);
                _mm512_storeu_ps(C + j + 32, sum2);
                _mm512_storeu_ps(C + j + 48, sum3);
            } else {
                // Fallback for remaining columns
                for (int64_t jj = j; jj < static_cast<int64_t>(N); jj += 16) {
                    __m512 sum_vec = _mm512_setzero_ps();
                    if (jj + 16 <= static_cast<int64_t>(N)) {
                        for (size_t k = 0; k < K; ++k) {
                            sum_vec = _mm512_fmadd_ps(_mm512_set1_ps(A[k]), _mm512_loadu_ps(B + k * ldb + jj), sum_vec);
                        }
                        _mm512_storeu_ps(C + jj, sum_vec);
                    } else {
                        // Scalar cleanup
                        for (size_t jjj = static_cast<size_t>(jj); jjj < N; ++jjj) {
                            float s = 0.0f;
                            for (size_t k = 0; k < K; ++k) s += A[k] * B[k * ldb + jjj];
                            C[jjj] = s;
                        }
                    }
                }
            }
        }
#else
        // Fallback to scalar (M=1)
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) sum += A[k] * B[k * ldb + j];
            C[j] = sum;
        }
#endif
        return;
    }

    // Tiling parameters could be larger for AVX-512
    const size_t MC = 128;
    const size_t NC = 128;
    const size_t KC = 256;

    // Zeroing C with OpenMP
    #pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(M); ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * ldc + j] = 0.0f;
        }
    }

    #pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(M); i += static_cast<int64_t>(MC)) {
        size_t m_end = std::min(i + MC, M);
        for (size_t j = 0; j < N; j += NC) {
            size_t n_end = std::min(j + NC, N);
            for (size_t k = 0; k < K; k += KC) {
                size_t k_end = std::min(k + KC, K);

                for (size_t ii = i; ii < m_end; ii += 16) {
                    for (size_t jj = j; jj < n_end; jj += 16) {
                        if (ii + 16 <= m_end && jj + 16 <= n_end) {
#if defined(__AVX512F__) || defined(_MSC_VER)
                            microkernel_16x16(A + ii * lda + k, B + k * ldb + jj, C + ii * ldc + jj, k_end - k, lda,
                                              ldb, ldc);
#else
                            // Scalar edge fallback
                            for (size_t iii = ii; iii < ii + 16; ++iii) {
                                for (size_t jjj = jj; jjj < jj + 16; ++jjj) {
                                    float sum = 0.0f;
                                    for (size_t kkk = k; kkk < k_end; ++kkk) {
                                        sum += A[iii * lda + kkk] * B[kkk * ldb + jjj];
                                    }
                                    C[iii * ldc + jjj] += sum;
                                }
                            }
#endif
                        } else {
                            // Boundary handling
                            size_t ii_end = std::min(ii + 16, m_end);
                            size_t jj_end = std::min(jj + 16, n_end);
                            for (size_t iii = ii; iii < ii_end; ++iii) {
                                for (size_t jjj = jj; jjj < jj_end; ++jjj) {
                                    float sum = 0.0f;
                                    for (size_t kkk = k; kkk < k_end; ++kkk) {
                                        sum += A[iii * lda + kkk] * B[kkk * ldb + jjj];
                                    }
                                    C[iii * ldc + jjj] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace blas
}  // namespace core
}  // namespace serius
