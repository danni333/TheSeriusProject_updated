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

}  // namespace blas
}  // namespace core
}  // namespace serius
