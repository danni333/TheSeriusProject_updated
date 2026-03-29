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
void fused_bias_gelu(void *x, const void *bias, size_t n, size_t x_dim, DType dtype, cudaStream_t stream = nullptr);

// RMSNorm: out = (x / RMS(x)) * weight
void rmsnorm(void *out, const void *input, const void *weight, size_t n, size_t batch_seq, float eps, DType dtype,
             cudaStream_t stream = 0);

// SwiGLU: out = silu(x) * y
void swiglu(void *out, const void *x, const void *y, size_t n, DType dtype, cudaStream_t stream = 0);

// RoPE (Rotary Positional Embedding)
void rope(void *data, size_t head_dim, size_t seq_len, size_t n_heads, float base, int pos_offset, DType dtype,
          cudaStream_t stream = 0);

// Softmax (row-wise)
void softmax(void *x, size_t rows, size_t cols, DType dtype, cudaStream_t stream = nullptr);

// --- Utility Kernels ---

/// out += other (element-wise)
void add_inplace(void *out, const void *other, size_t n, DType dtype, cudaStream_t stream = nullptr);

/// out[i, :] = table[indices[i], :]
void gather(void *out, const void *table, const int *indices, size_t n, size_t dim, DType dtype,
            cudaStream_t stream = nullptr);

// Repeat KV: expands n_kv heads to n_heads (GQA)
// x: input tensor (batch_seq, n_kv, head_dim)
// out: output tensor (batch_seq, n_heads, head_dim)
void repeat_kv(void *out, const void *x, size_t batch_seq, size_t n_heads, size_t n_kv, size_t head_dim, 
               DType dtype, cudaStream_t stream = 0);

/// Copy from src to dst with a layout transformation defined by src_strides and shape.
/// Used for GPU-native contiguous() and transpose() operations.
void copy_layout(void *dst, const void *src, const size_t *shape, const size_t *src_strides, size_t dims,
                 DType dtype, cudaStream_t stream = 0);

}  // namespace hal
}  // namespace serius
