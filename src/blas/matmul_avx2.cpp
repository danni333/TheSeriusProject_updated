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
    // Tiling parameters
    const size_t MC = 64;
    const size_t NC = 64;
    const size_t KC = 256;

    // Zero the C matrix first, as our microkernel accumulates
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * ldc + j] = 0.0f;
        }
    }

    for (size_t i = 0; i < M; i += MC) {
        size_t m_end = std::min(i + MC, M);
        for (size_t j = 0; j < N; j += NC) {
            size_t n_end = std::min(j + NC, N);
            for (size_t k = 0; k < K; k += KC) {
                size_t k_end = std::min(k + KC, K);

                // For the inner blocks, we can use our 8x8 microkernel if dimensions allow
                for (size_t ii = i; ii < m_end; ii += 8) {
                    for (size_t jj = j; jj < n_end; jj += 8) {
                        if (ii + 8 <= m_end && jj + 8 <= n_end) {
                            microkernel_8x8(A + ii * lda + k, B + k * ldb + jj, C + ii * ldc + jj, k_end - k, lda, ldb,
                                            ldc);
                        } else {
                            // Boundary handling (scalar fallback for edges)
                            size_t ii_end = std::min(ii + 8, m_end);
                            size_t jj_end = std::min(jj + 8, n_end);
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
