#pragma once

#include "../tensor/tensor.h"

namespace serius {
namespace core {
namespace blas {

// --- Matrix Multiplication ---

// C = A * B
void matmul(const Tensor &A, const Tensor &B, Tensor &C, void *stream = nullptr);

// y = A * x
void matvec(const Tensor &A, const Tensor &x, Tensor &y);

// C = A * B (Batched). A is (B, M, K), B is (B, K, N) or (K, N)
void batched_matmul(const Tensor &A, const Tensor &B, Tensor &C, void *stream = nullptr);

// --- Element-wise Operations ---

// C = A + B
void element_wise_add(const Tensor &A, const Tensor &B, Tensor &C);

// C = A - B
void element_wise_sub(const Tensor &A, const Tensor &B, Tensor &C);

// C = A * B (element-wise multiplication)
void element_wise_mul(const Tensor &A, const Tensor &B, Tensor &C);

// C = A / B (element-wise division)
void element_wise_div(const Tensor &A, const Tensor &B, Tensor &C);

// B = alpha * A
void scale(const Tensor &A, float alpha, Tensor &B);

// Apply a function element-wise
void element_wise_apply(const Tensor &A, float (*func)(float), Tensor &B);

// --- Advanced/Internal Declarations ---
// These are declarations for internal dispatch engines

void matmul_scalar(const float *A, const float *B, float *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                   size_t ldc);
void matmul_avx2(const float *A, const float *B, float *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                 size_t ldc);
void matmul_avx512(const float *A, const float *B, float *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                   size_t ldc);

// CPU capabilities
struct CpuInfo
{
    bool has_avx2;
    bool has_avx512f;
};

CpuInfo get_cpu_info();

// --- Convenience Wrappers ---
Tensor matmul(const Tensor &A, const Tensor &B, void *stream = nullptr);
Tensor batched_matmul(const Tensor &A, const Tensor &B, void *stream = nullptr);
Tensor matvec(const Tensor &A, const Tensor &x);
Tensor element_wise_add(const Tensor &A, const Tensor &B);
Tensor element_wise_sub(const Tensor &A, const Tensor &B);
Tensor element_wise_mul(const Tensor &A, const Tensor &B);
Tensor element_wise_div(const Tensor &A, const Tensor &B);
Tensor scale(const Tensor &A, float alpha);
Tensor element_wise_apply(const Tensor &A, float (*func)(float));

}  // namespace blas
}  // namespace core
}  // namespace serius
