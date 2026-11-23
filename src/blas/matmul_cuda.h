#pragma once

#include "../hal/device.h"

namespace serius {
namespace core {
namespace blas {

void matmul_cuda(const void *A, const void *B, void *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                 size_t ldc, hal::DType dtype, void *stream = nullptr);

void batched_matmul_cuda(const void *A, const void *B, void *C, size_t batch_size, size_t M, size_t N, size_t K,
                         size_t lda, size_t ldb, size_t ldc, size_t stride_a, size_t stride_b, size_t stride_c,
                         hal::DType dtype, void *stream = nullptr);

/// logits[s, v] = sum_h x[s, h] * w[v, h]. w is FP16 [vocab, hidden] row-major,
/// x is FP16 [seq, hidden] row-major, out is FP32 [seq, vocab] row-major.
/// Implemented as a single cuBLAS GemmEx with FP32 accumulation.
void logits_gemv_mixed_cublas(const void *x, const void *w, void *out, size_t seq, size_t vocab, size_t hidden,
                              void *stream = nullptr);

/// Tiled variant of logits_gemv_mixed_cublas: computes vocab rows
/// [v0, v0+n) of out[seq, vocab_total], where `w` is the FP16 [n, hidden]
/// tile holding exactly those rows (e.g. a dequantized slice of the packed
/// lm_head). Tiles cover disjoint columns of an uninitialized `out`, so
/// `beta` should be 0.
void logits_gemm_mixed_cublas(const void *x, const void *w, void *out, size_t seq, size_t vocab_total, size_t v0,
                              size_t n, size_t hidden, float beta, void *stream = nullptr);

}  // namespace blas
}  // namespace core
}  // namespace serius
