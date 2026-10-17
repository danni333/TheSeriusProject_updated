#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace serius {
namespace hal {

namespace {
constexpr float SOFTMAX_EPS = 1e-6F;
constexpr int D_32 = 32;
constexpr int D_64 = 64;
constexpr int D_128 = 128;

void check_cuda_status(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
}  // namespace

/**
 * @brief Flash Attention v2 Forward Kernel
 */
template <int HEAD_DIM>
__global__ void flash_attn_fwd_kernel(float *__restrict__ output, float *__restrict__ lse,
                                      const float *__restrict__ query, const float *__restrict__ key,
                                      const float *__restrict__ value, int seq_len_q, int seq_len_k,
                                      float softmax_scale, bool is_causal, int block_m, int block_n)
{
    const int q_block_idx = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch_id = blockIdx.z;
    const int tid = threadIdx.x;

    const int head_offset = (batch_id * gridDim.y + head_id) * seq_len_q * HEAD_DIM;
    const int kv_head_offset = (batch_id * gridDim.y + head_id) * seq_len_k * HEAD_DIM;
    const int lse_head_offset = (batch_id * gridDim.y + head_id) * seq_len_q;

    const float *query_head = query + head_offset;
    const float *key_head = key + kv_head_offset;
    const float *value_head = value + kv_head_offset;
    float *output_head = output + head_offset;
    float *lse_head = lse + lse_head_offset;

    extern __shared__ float shared_memory[];
    float *s_query = shared_memory;
    float *s_key = s_query + block_m * HEAD_DIM;
    float *s_value = s_key + block_n * HEAD_DIM;

    float acc[HEAD_DIM];
#pragma unroll
    for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
        acc[dim_i] = 0.0F;
    }
    float l_i = 0.0F;
    float m_i = -FLT_MAX;

    const int q_idx = (q_block_idx * block_m) + tid;

    if (q_idx < seq_len_q) {
#pragma unroll
        for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
            s_query[(tid * HEAD_DIM) + dim_i] = query_head[(q_idx * HEAD_DIM) + dim_i];
        }
    }
    __syncthreads();

    const int num_n_blocks = (seq_len_k + block_n - 1) / block_n;
    for (int n_block_idx = 0; n_block_idx < num_n_blocks; ++n_block_idx) {
        const int k_start = n_block_idx * block_n;

        for (int i_idx = tid; i_idx < block_n; i_idx += block_m) {
            int k_idx = k_start + i_idx;
            if (k_idx < seq_len_k) {
#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    s_key[(i_idx * HEAD_DIM) + dim_i] = key_head[(k_idx * HEAD_DIM) + dim_i];
                    s_value[(i_idx * HEAD_DIM) + dim_i] = value_head[(k_idx * HEAD_DIM) + dim_i];
                }
            } else {
#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    s_key[(i_idx * HEAD_DIM) + dim_i] = 0.0F;
                    s_value[(i_idx * HEAD_DIM) + dim_i] = 0.0F;
                }
            }
        }
        __syncthreads();

        if (q_idx < seq_len_q) {
            for (int j_idx = 0; j_idx < block_n; ++j_idx) {
                const int k_idx = k_start + j_idx;
                if (k_idx >= seq_len_k || (is_causal && q_idx < k_idx)) {
                    continue;
                }

                float s_ij = 0.0F;
#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    s_ij += s_query[(tid * HEAD_DIM) + dim_i] * s_key[(j_idx * HEAD_DIM) + dim_i];
                }
                s_ij *= softmax_scale;

                float m_new = max(m_i, s_ij);
                float exp_prev = expf(m_i - m_new);
                float exp_curr = expf(s_ij - m_new);

                l_i = (l_i * exp_prev) + exp_curr;
#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    acc[dim_i] = (acc[dim_i] * exp_prev) + (exp_curr * s_value[(j_idx * HEAD_DIM) + dim_i]);
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (q_idx < seq_len_q) {
        float inv_l = 1.0F / (l_i + SOFTMAX_EPS);
#pragma unroll
        for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
            output_head[(q_idx * HEAD_DIM) + dim_i] = acc[dim_i] * inv_l;
        }
        lse_head[q_idx] = m_i + logf(l_i + SOFTMAX_EPS);
    }
}

/**
 * @brief Flash Attention v2 Backward Kernel
 */
template <int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__global__ void flash_attn_bwd_kernel(float *__restrict__ dQ, float *__restrict__ dK, float *__restrict__ dV,
                                      const float *__restrict__ Q, const float *__restrict__ K,
                                      const float *__restrict__ V, const float *__restrict__ O,
                                      const float *__restrict__ dO, const float *__restrict__ LSE, int seq_len_q,
                                      int seq_len_k, float softmax_scale, bool is_causal)
{
    const int br = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch_id = blockIdx.z;
    const int tid = threadIdx.x;

    const int q_offset = (batch_id * gridDim.y + head_id) * seq_len_q * HEAD_DIM;
    const int kv_offset = (batch_id * gridDim.y + head_id) * seq_len_k * HEAD_DIM;
    const int lse_offset = (batch_id * gridDim.y + head_id) * seq_len_q;

    const float *Q_head = Q + q_offset;
    const float *K_head = K + kv_offset;
    const float *V_head = V + kv_offset;
    const float *O_head = O + q_offset;
    const float *dO_head = dO + q_offset;
    const float *LSE_head = LSE + lse_offset;

    float *dQ_head = dQ + q_offset;
    float *dK_head = dK + kv_offset;
    float *dV_head = dV + kv_offset;

    __shared__ float s_Q[BLOCK_M * HEAD_DIM];
    __shared__ float s_K[BLOCK_N * HEAD_DIM];
    __shared__ float s_V[BLOCK_N * HEAD_DIM];
    __shared__ float s_dO[BLOCK_M * HEAD_DIM];
    __shared__ float s_LSE[BLOCK_M];

    const int q_idx = (br * BLOCK_M) + tid;

    if (q_idx < seq_len_q) {
#pragma unroll
        for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
            s_Q[(tid * HEAD_DIM) + dim_i] = Q_head[(q_idx * HEAD_DIM) + dim_i];
            s_dO[(tid * HEAD_DIM) + dim_i] = dO_head[(q_idx * HEAD_DIM) + dim_i];
        }
        s_LSE[tid] = LSE_head[q_idx];
    }
    __syncthreads();

    float dq_acc[HEAD_DIM];
#pragma unroll
    for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
        dq_acc[dim_i] = 0.0F;
    }

    const int num_n_blocks = (seq_len_k + BLOCK_N - 1) / BLOCK_N;
    for (int bc = 0; bc < num_n_blocks; ++bc) {
        const int k_start = bc * BLOCK_N;

        for (int i_idx = tid; i_idx < BLOCK_N; i_idx += BLOCK_M) {
            int k_idx = k_start + i_idx;
            if (k_idx < seq_len_k) {
#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    s_K[(i_idx * HEAD_DIM) + dim_i] = K_head[(k_idx * HEAD_DIM) + dim_i];
                    s_V[(i_idx * HEAD_DIM) + dim_i] = V_head[(k_idx * HEAD_DIM) + dim_i];
                }
            } else {
#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    s_K[(i_idx * HEAD_DIM) + dim_i] = 0.0F;
                    s_V[(i_idx * HEAD_DIM) + dim_i] = 0.0F;
                }
            }
        }
        __syncthreads();

        if (q_idx < seq_len_q) {
            for (int j_idx = 0; j_idx < BLOCK_N; ++j_idx) {
                const int k_idx = k_start + j_idx;
                if (k_idx >= seq_len_k || (is_causal && q_idx < k_idx)) continue;

                float s_ij = 0.0F;
#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    s_ij += s_Q[(tid * HEAD_DIM) + dim_i] * s_K[(j_idx * HEAD_DIM) + dim_i];
                }
                s_ij *= softmax_scale;

                float p_ij = expf(s_ij - s_LSE[tid]);

#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    atomicAdd(&dV_head[(k_idx * HEAD_DIM) + dim_i], p_ij * s_dO[(tid * HEAD_DIM) + dim_i]);
                }

                float dp_ij_sum = 0.0F;
#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    dp_ij_sum += s_dO[(tid * HEAD_DIM) + dim_i] * s_V[(j_idx * HEAD_DIM) + dim_i];
                }

                float l_grad_sum = 0.0F;
#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    l_grad_sum += s_dO[(tid * HEAD_DIM) + dim_i] * O_head[(q_idx * HEAD_DIM) + dim_i];
                }

                float dS_ij = p_ij * (dp_ij_sum - l_grad_sum);

#pragma unroll
                for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
                    dq_acc[dim_i] += (dS_ij * softmax_scale) * s_K[(j_idx * HEAD_DIM) + dim_i];
                    atomicAdd(&dK_head[(k_idx * HEAD_DIM) + dim_i],
                              (dS_ij * softmax_scale) * s_Q[(tid * HEAD_DIM) + dim_i]);
                }
            }
        }
        __syncthreads();
    }

    if (q_idx < seq_len_q) {
#pragma unroll
        for (int dim_i = 0; dim_i < HEAD_DIM; ++dim_i) {
            atomicAdd(&dQ_head[(q_idx * HEAD_DIM) + dim_i], dq_acc[dim_i]);
        }
    }
}

// Host dispatch
void flash_attention_v2_fwd_cuda(float *output, float *lse, const float *query, const float *key, const float *value,
                                 int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim,
                                 float softmax_scale, bool is_causal, int block_m, int block_n)
{
    // Validate input parameters to prevent undefined behavior
    if (!output || !lse || !query || !key || !value) {
        throw std::invalid_argument("Flash attention forward: null device pointer");
    }

    if (batch_size <= 0 || num_heads <= 0 || seq_len_q <= 0 || seq_len_k <= 0 || head_dim <= 0) {
        throw std::invalid_argument("Flash attention forward: dimensions must be positive");
    }

    // Validate that head_dim is one of the supported values
    if (head_dim != D_32 && head_dim != D_64 && head_dim != D_128) {
        throw std::invalid_argument("Flash attention forward: unsupported head dimension");
    }

    const auto is_supported_block = [](int block) { return block == 16 || block == 32 || block == 64 || block == 128; };
    if (!is_supported_block(block_m) || !is_supported_block(block_n)) {
        throw std::invalid_argument("Flash attention forward: block sizes must be 16, 32, 64, or 128");
    }
    const size_t shared_memory_bytes =
        (static_cast<size_t>(block_m) + 2 * static_cast<size_t>(block_n)) * static_cast<size_t>(head_dim) *
        sizeof(float);
    if (shared_memory_bytes > 48 * 1024) {
        throw std::invalid_argument("Flash attention forward: blocks exceed the 48 KiB shared-memory limit");
    }
    dim3 grid((seq_len_q + block_m - 1) / block_m, num_heads, batch_size);
    if (grid.y > 65535 || grid.z > 65535) {
        throw std::invalid_argument("Flash attention forward: CUDA grid too large");
    }
    if (head_dim == D_32) {
        flash_attn_fwd_kernel<32><<<grid, block_m, shared_memory_bytes>>>(
            output, lse, query, key, value, seq_len_q, seq_len_k, softmax_scale, is_causal, block_m, block_n);
    } else if (head_dim == D_64) {
        flash_attn_fwd_kernel<64><<<grid, block_m, shared_memory_bytes>>>(
            output, lse, query, key, value, seq_len_q, seq_len_k, softmax_scale, is_causal, block_m, block_n);
    } else {
        flash_attn_fwd_kernel<128><<<grid, block_m, shared_memory_bytes>>>(
            output, lse, query, key, value, seq_len_q, seq_len_k, softmax_scale, is_causal, block_m, block_n);
    }
    check_cuda_status(cudaGetLastError(), "Flash attention forward kernel launch");
}

void flash_attention_v2_bwd_cuda(float *dq, float *dk, float *dv, const float *q, const float *k, const float *v,
                                 const float *o, const float *do_grad, const float *lse, int batch_size, int num_heads,
                                 int seq_len_q, int seq_len_k, int head_dim, float softmax_scale, bool is_causal)
{
    if (!dq || !dk || !dv || !q || !k || !v || !o || !do_grad || !lse) {
        throw std::invalid_argument("Flash attention backward: null device pointer");
    }
    if (batch_size <= 0 || num_heads <= 0 || seq_len_q <= 0 || seq_len_k <= 0) {
        throw std::invalid_argument("Flash attention backward: dimensions must be positive");
    }
    if (head_dim != D_32 && head_dim != D_64 && head_dim != D_128) {
        throw std::invalid_argument("Flash attention backward: unsupported head dimension");
    }

    size_t size_q = (size_t)batch_size * num_heads * seq_len_q * head_dim;
    size_t size_kv = (size_t)batch_size * num_heads * seq_len_k * head_dim;
    check_cuda_status(cudaMemset(dq, 0, size_q * sizeof(float)), "Flash attention clear dQ");
    check_cuda_status(cudaMemset(dk, 0, size_kv * sizeof(float)), "Flash attention clear dK");
    check_cuda_status(cudaMemset(dv, 0, size_kv * sizeof(float)), "Flash attention clear dV");

    if (head_dim == D_32) {
        // Reduced block size to fit shared memory (64*32*4*4 = 32KB < 48KB)
        dim3 grid((seq_len_q + 64 - 1) / 64, num_heads, batch_size);
        flash_attn_bwd_kernel<32, 64, 64>
            <<<grid, 64>>>(dq, dk, dv, q, k, v, o, do_grad, lse, seq_len_q, seq_len_k, softmax_scale, is_causal);
    } else if (head_dim == D_64) {
        // Reduced block size (32*64*4*4 = 32KB < 48KB)
        dim3 grid((seq_len_q + 32 - 1) / 32, num_heads, batch_size);
        flash_attn_bwd_kernel<64, 32, 32>
            <<<grid, 32>>>(dq, dk, dv, q, k, v, o, do_grad, lse, seq_len_q, seq_len_k, softmax_scale, is_causal);
    } else if (head_dim == D_128) {
        // Reduced block size (16*128*4*4 = 32KB < 48KB)
        dim3 grid((seq_len_q + 16 - 1) / 16, num_heads, batch_size);
        flash_attn_bwd_kernel<128, 16, 16>
            <<<grid, 16>>>(dq, dk, dv, q, k, v, o, do_grad, lse, seq_len_q, seq_len_k, softmax_scale, is_causal);
    }
    check_cuda_status(cudaGetLastError(), "Flash attention backward kernel launch");
}

}  // namespace hal
}  // namespace serius
