#pragma once

#ifdef SERIUS_USE_CUDA
#include <cuda_runtime.h>
#else
typedef void *cudaStream_t;
#endif

#include "device.h"

namespace serius {
namespace hal {

// Fused LayerNorm(x + residual)
// x: input tensor (also receives the result if in-place)
// residual: residual connection to be added
// gamma, beta: scale and shift parameters
// n: total elements in the hidden dimension (d_model)
// batch_seq: number of rows (batch_size * seq_len)
void fused_layernorm_residual(void *x, const void *residual, const void *gamma, const void *beta, size_t n,
                              size_t batch_seq, float eps, DType dtype, cudaStream_t stream = 0);

// Fused Masked Softmax for Attention
// scores: attention scores (batch, heads, seq_q, seq_k)
// mask: causal mask (optional, if nullptr no masking is applied)
// batch_heads: batch_size * n_heads
// seq_q, seq_k: sequence lengths
void fused_masked_softmax(void *x, const void *mask, size_t batch_heads, size_t seq_q, size_t seq_k, DType dtype,
                          float scale, cudaStream_t stream = 0);

// Fused Bias + GELU activation
// x: input tensor (receives result in-place)
// bias: bias vector (size = hidden_dim, broadcasted across batch)
// n: total number of elements
// bias_size: size of the bias vector (hidden_dim)
void fused_bias_gelu(void *x, const void *bias, size_t n, size_t bias_size, DType dtype, cudaStream_t stream = nullptr);

}  // namespace hal
}  // namespace serius
