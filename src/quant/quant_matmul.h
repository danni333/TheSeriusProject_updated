#pragma once

// Shared naive dequant-matmul loop for the quantized weight formats
// (Q4/INT4, FP8, AWQ, GPTQ). Each format supplies a decode(j, k) functor
// returning the dequantized weight W[j, k]; the accumulation order is
// identical to the previous per-format loops, so results are bit-identical.

#include <cstddef>

namespace serius {
namespace quant {

/// Y[M,N] = X[M,K] * W[N,K]^T, W dequantized on the fly via decode(j, k).
template <typename DecodeFn>
inline void dequant_matmul_t(const float *X, float *Y, int M, int N, int K, DecodeFn &&decode)
{
    for (int i = 0; i < M; ++i) {
        const float *x_row = X + static_cast<size_t>(i) * K;
        float *y_row = Y + static_cast<size_t>(i) * N;
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += x_row[k] * decode(j, k);
            }
            y_row[j] = sum;
        }
    }
}

}  // namespace quant
}  // namespace serius
