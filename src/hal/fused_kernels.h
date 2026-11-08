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

/// In-place multiply by a scalar. Used to pre-scale Q by 1/sqrt(head_dim)
/// so attention scores stay within reduced-precision range.
void scale_inplace(void *data, size_t n, float s, DType dtype, cudaStream_t stream = nullptr);

/// RMSNorm with FP32 input and reduced-precision weight/output.
/// Used to keep the residual stream in FP32 while weights stay FP16/BF16.
void rmsnorm_mixed(void *out, const void *input, const void *weight, size_t n, size_t batch_seq, float eps,
                   DType out_dtype, cudaStream_t stream = nullptr);

/// Gather rows from a reduced-precision table into an FP32 output.
void gather_mixed(void *out, const void *table, const int *indices, size_t n, size_t dim, DType table_dtype,
                  DType out_dtype, cudaStream_t stream = nullptr);

/// out += other with FP32 accumulator and reduced-precision input.
void add_inplace_mixed(void *out, const void *other, size_t n, DType acc_dtype, DType in_dtype,
                       cudaStream_t stream = nullptr);

/// GEMV for the M==1 decode path: y[1,N] = x[1,K] @ W[K,N] where W is the
/// row-major {in_features, out_features} projection weight. Streams W once.
void gemv_kn(void *x, const void *w, void *y, size_t K, size_t N, DType dtype,
             cudaStream_t stream = nullptr);

/// Split-K FP16 GEMV: accumulates into y_f32_acc (must be zeroed, N floats),
/// then the caller converts to the output dtype. For M==1 decode projections.
void gemv_kn_sk_f16(const void *x, const void *w, void *y_f32_acc, size_t K, size_t N,
                    cudaStream_t stream = nullptr);

/// Quantized GEMV: y[row] = dot(x, dequant(W_row)) for GGUF Q4_0/Q4_1/Q8_0.
/// x is FP16 [in], w is packed QuantWeight bytes [out rows x row_bytes],
/// y_f32 is a zeroed FP32 [out] accumulator (caller converts/syncs).
/// qtype: 3=Q4_0, 4=Q4_1, 5=Q8_0 (quant::QuantType values).
void gemv_quant_f16(const void *x, const void *w, int qtype, size_t in, size_t out, void *y_f32,
                    cudaStream_t stream = nullptr);

/// Warp-per-row quantized GEMV: x must be FP32 [in] (pre-converted once per
/// projection input), y_f32 is FP32 [out] written directly (no zeroing needed).
/// Same qtype encoding as gemv_quant_f16. Much faster than the split-K variant
/// for large `out` (coalesced packed-weight reads, no atomics).
void gemv_quant_f32(const void *x_f32, const void *w, int qtype, size_t in, size_t out, void *y_f32,
                    cudaStream_t stream = nullptr);

/// Vectorized variant over QuantWeight::dev_bytes (repacked layout): per row,
/// block-major quant payload (16B vector loads), then fp16 d/m scale arrays.
/// kind/stride/qs/d_off/m_off come from the QuantWeight dev_* fields. Same
/// warp-per-row mapping; fastest path for Q4_0/Q4_1/Q8_0 decode.
/// `sx16` (required for kind 3/4, may be null for kind 5) holds per-16-element
/// sums of x: sx16[i] = sum(x[16i .. 16i+15]), length in/16. Used to fold the
/// Q4_0/Q4_1 affine terms into O(1) per-block corrections.
void gemv_quant_vec_f32(const void *x_f32, const void *sx16, const void *w, int kind,
                        size_t in, size_t out, int stride, int qs, int d_off, int m_off,
                        void *y_f32, cudaStream_t stream = nullptr);

/// sx16[i] = sum(x[16i .. 16i+15]) for i < n16 (= in/16). One float4 x4 per
/// thread; used by gemv_quant_vec_f32 for Q4_0/Q4_1.
void block_sums16_f32(const void *x_f32, void *sx16, int n16, cudaStream_t stream = nullptr);

/// GPU top-k preselection over a FP32 logits row [vocab]. Returns up to `cap`
/// candidate (index, logit) pairs whose logit exceeds max_logit - window,
/// widening the window until at least `need` candidates are found (or `cap`
/// is hit). `count` receives the number of stored candidates; the true top-k
/// is guaranteed to be inside iff count >= need. idx/val are host arrays.
/// Returns false on CUDA error or if fewer than `need` candidates exist even
/// at the widest window — caller should fall back to CPU sampling.
int topk_logits_f32(const void *logits_gpu, int vocab, int need, int cap,
                    int *idx_out, float *val_out, cudaStream_t stream = nullptr);

/// Device-side dtype conversion between FP32/FP16/BF16 (no host round-trip).
void convert_dtype(void *out, const void *in, size_t n, DType in_dtype, DType out_dtype,
                   cudaStream_t stream = nullptr);

/// logits[s, v] = sum_h x[s, h] * w_t[v, h]. w_t must be [vocab, hidden] FP32
/// row-major (i.e. the transposed lm_head). Coalesced, ~memory-bound.
void logits_gemv_f32(const void *x, const void *w_t, void *out, size_t seq, size_t vocab, size_t hidden,
                     cudaStream_t stream = nullptr);

// RoPE (Rotary Positional Embedding)
void rope(void *data, size_t head_dim, size_t seq_len, size_t n_heads, float base, int pos_offset, DType dtype,
          cudaStream_t stream = 0);

/// RoPE variant for CUDA Graph capture: pos_offset is read from device memory
/// (*pos_dev) instead of a kernel argument, so one captured graph works for
/// every decode position. Only seq_len==1 is supported (pos = *pos_dev).
void rope_devpos(void *data, size_t head_dim, size_t seq_len, size_t n_heads, float base,
                 const int *pos_dev, DType dtype, cudaStream_t stream = 0);

// Softmax (row-wise)
void softmax(void *x, size_t rows, size_t cols, DType dtype, cudaStream_t stream = nullptr);

// --- Utility Kernels ---

/// out += other (element-wise)
void add_inplace(void *out, const void *other, size_t n, DType dtype, cudaStream_t stream = nullptr);

/// out[r, c] += bias[c] for a [rows, cols] matrix (broadcast bias over rows)
void add_row_bias(void *out, const void *bias, size_t rows, size_t cols, DType dtype,
                  cudaStream_t stream = nullptr);

/// out[i, :] = table[indices[i], :]
void gather(void *out, const void *table, const int *indices, size_t n, size_t dim, DType dtype,
            cudaStream_t stream = nullptr);

// Repeat KV: expands n_kv heads to n_heads (GQA)
// x: input tensor (batch_seq, n_kv, head_dim)
// out: output tensor (batch_seq, n_heads, head_dim)
void repeat_kv(void *out, const void *x, size_t batch_seq, size_t n_heads, size_t n_kv, size_t head_dim, 
               DType dtype, cudaStream_t stream = 0);

/// Fused single-token GQA decode attention (FP16 only): one kernel replaces
/// repeat_kv + transposes + batched_matmul + softmax. q is [n_heads*head_dim]
/// pre-scaled by 1/sqrt(head_dim); K/V are [max_seq, kv_dim] cache rows;
/// out is [n_heads*head_dim]. pos = valid cache rows (must be <= 4096).
void gqa_decode_attention_f16(const void *q, const void *K, const void *V, void *out,
                              int pos, int n_heads, int n_kv, int head_dim, int kv_dim,
                              cudaStream_t stream = 0);

/// Device-position variant for CUDA Graph capture: valid rows = (*pos_dev) + 1.
/// Shared memory is allocated for the maximum 4096 positions, so pos_dev must
/// never exceed 4095. All other arguments identical to gqa_decode_attention_f16.
void gqa_decode_attention_f16_devpos(const void *q, const void *K, const void *V, void *out,
                                     const int *pos_dev, int n_heads, int n_kv, int head_dim,
                                     int kv_dim, cudaStream_t stream = 0);

/// Copy src[0..nbytes) to dst_base + (*pos_dev) * row_stride_bytes.
/// Byte-granular; used to write the new K/V row into the cache at a
/// device-side position (CUDA Graph safe — no host-computed offset).
void kv_write_devpos(void *dst_base, const void *src, size_t nbytes,
                     const int *pos_dev, size_t row_stride_bytes,
                     cudaStream_t stream = 0);

/// Copy from src to dst with a layout transformation defined by src_strides and shape.
/// Used for GPU-native contiguous() and transpose() operations.
void copy_layout(void *dst, const void *src, const size_t *shape, const size_t *src_strides, size_t dims,
                 DType dtype, cudaStream_t stream = 0);

}  // namespace hal
}  // namespace serius
