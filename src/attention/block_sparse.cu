#include "sparsity_mask.h"

#ifdef SERIUS_USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace serius {
namespace attention {

namespace {

constexpr int kThreads = 256;
// Finite stand-in for -inf: expf(-1e30f - m) underflows to 0 and fmaxf
// comparisons stay NaN-free.
constexpr float kNegInf = -1e30f;

// Block-sparse attention with proper softmax, one thread block per query
// row. For each query row we iterate over the k-blocks enabled by
// block_mask and maintain an online softmax (running max m, sum-exp l and
// accumulator acc[d]) in shared memory, rescaling whenever the max grows —
// the same numerically-stable scheme as flash attention, but restricted to
// the active blocks.
//
// Shared memory layout (floats):
//   [0, block_size)          scores/weights for the current k-block
//   [block_size, +d_model)   per-dimension output accumulator
//   [+40]                    reduction scratch + {bmax, m, l}
__global__ void block_sparse_attention_kernel(
    const float* Q, const float* K, const float* V, float* O,
    const int* block_mask, int seq_len, int d_model, int block_size,
    int num_blocks, float scale)
{
    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_acc = smem + block_size;
    float* s_red = s_acc + d_model;  // 40 floats: [0..31] warp partials,
                                     // [32]=bmax, [33]=m, [34]=l

    const int tid = threadIdx.x;
    const int q = blockIdx.x;
    const int q_block = q / block_size;
    const float* q_row = Q + static_cast<size_t>(q) * d_model;

    if (tid == 0) {
        s_red[32] = kNegInf;  // block max
        s_red[33] = kNegInf;  // running max
        s_red[34] = 0.0f;     // running sum-exp
    }
    for (int d = tid; d < d_model; d += kThreads) {
        s_acc[d] = 0.0f;
    }
    __syncthreads();

    for (int kb = 0; kb < num_blocks; ++kb) {
        if (block_mask[q_block * num_blocks + kb] == 0) {
            continue;
        }
        const int k_start = kb * block_size;
        const int k_len = min(block_size, seq_len - k_start);

        // Phase 1: scores s_j = scale * <q, k_j> and block max.
        // Keys past the query position stay masked (causal attention).
        float lmax = kNegInf;
        for (int j = tid; j < k_len; j += kThreads) {
            const int k_idx = k_start + j;
            float s = kNegInf;
            if (k_idx <= q) {
                const float* k_row = K + static_cast<size_t>(k_idx) * d_model;
                float dot = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    dot += q_row[d] * k_row[d];
                }
                s = dot * scale;
            }
            s_scores[j] = s;
            lmax = fmaxf(lmax, s);
        }
        // Warp + block reduction for the max.
        for (int off = 16; off > 0; off >>= 1) {
            lmax = fmaxf(lmax, __shfl_down_sync(0xffffffffu, lmax, off));
        }
        if ((tid & 31) == 0) {
            s_red[tid >> 5] = lmax;
        }
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (kThreads >> 5)) ? s_red[tid] : kNegInf;
            for (int off = 16; off > 0; off >>= 1) {
                v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
            }
            if (tid == 0) {
                s_red[32] = v;
            }
        }
        __syncthreads();

        const float bmax = s_red[32];
        if (bmax <= kNegInf * 0.5f) {
            continue;  // every key in this block is masked
        }
        const float m_old = s_red[33];
        const float m_new = fmaxf(m_old, bmax);

        // Phase 2: rescale accumulator and sum-exp to the new max.
        const float corr = expf(m_old - m_new);  // kNegInf -> 0 on first block
        for (int d = tid; d < d_model; d += kThreads) {
            s_acc[d] *= corr;
        }
        if (tid == 0) {
            s_red[34] *= corr;
            s_red[33] = m_new;
        }
        __syncthreads();

        // Phase 3: weights w_j = exp(s_j - m_new); add their sum to l.
        float wpart = 0.0f;
        for (int j = tid; j < k_len; j += kThreads) {
            const float w = expf(s_scores[j] - m_new);
            s_scores[j] = w;
            wpart += w;
        }
        for (int off = 16; off > 0; off >>= 1) {
            wpart += __shfl_down_sync(0xffffffffu, wpart, off);
        }
        if ((tid & 31) == 0) {
            s_red[tid >> 5] = wpart;
        }
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (kThreads >> 5)) ? s_red[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            if (tid == 0) {
                s_red[34] += v;
            }
        }
        __syncthreads();

        // Phase 4: acc[d] += sum_j w_j * V[k_start + j, d].
        for (int d = tid; d < d_model; d += kThreads) {
            float a = 0.0f;
            for (int j = 0; j < k_len; ++j) {
                a += s_scores[j] * V[static_cast<size_t>(k_start + j) * d_model + d];
            }
            s_acc[d] += a;
        }
        __syncthreads();
    }

    const float l = s_red[34];
    const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int d = tid; d < d_model; d += kThreads) {
        O[static_cast<size_t>(q) * d_model + d] = s_acc[d] * inv_l;
    }
}

}  // namespace

Tensor block_sparse_attention_cuda(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    const Tensor& block_mask, size_t block_size, cudaStream_t stream)
{
    if (Q.shape().size() != 2 || K.shape() != Q.shape() ||
        V.shape() != Q.shape()) {
        throw std::runtime_error(
            "block_sparse_attention_cuda: Q, K, V must be 2-D tensors with "
            "identical shapes");
    }
    if (Q.dtype() != hal::DType::Float32 || block_size == 0) {
        throw std::runtime_error(
            "block_sparse_attention_cuda: float32 tensors and block_size > 0 "
            "required");
    }

    const size_t seq_len = Q.shape()[0];
    const size_t d_model = Q.shape()[1];
    const size_t num_blocks = (seq_len + block_size - 1) / block_size;
    if (block_mask.size() < num_blocks * num_blocks) {
        throw std::runtime_error(
            "block_sparse_attention_cuda: block_mask too small for "
            "seq_len/block_size");
    }

    // Convert block_mask to integer array for GPU (row-major, 1 = active).
    std::vector<int> mask_int(num_blocks * num_blocks);
    for (size_t i = 0; i < num_blocks * num_blocks; ++i) {
        mask_int[i] = (block_mask.item(i) > 0.5f) ? 1 : 0;
    }

    int* d_block_mask = nullptr;
    cudaMalloc(&d_block_mask, mask_int.size() * sizeof(int));
    cudaMemcpy(d_block_mask, mask_int.data(), mask_int.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    Tensor output({seq_len, d_model});
    Q.syncToDevice();
    K.syncToDevice();
    V.syncToDevice();
    output.syncToDevice();

    const size_t smem_floats = block_size + d_model + 40;
    const size_t smem_bytes = smem_floats * sizeof(float);

    static int max_smem = -1;
    if (max_smem < 0) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        max_smem = static_cast<int>(prop.sharedMemPerBlockOptin);
    }
    if (static_cast<int>(smem_bytes) > max_smem) {
        cudaFree(d_block_mask);
        throw std::runtime_error(
            "block_sparse_attention_cuda: shared memory requirement (" +
            std::to_string(smem_bytes) + " bytes) exceeds device limit (" +
            std::to_string(max_smem) + " bytes)");
    }
    const cudaError_t attr_err = cudaFuncSetAttribute(
        block_sparse_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (attr_err != cudaSuccess) {
        cudaFree(d_block_mask);
        throw std::runtime_error(std::string("block_sparse_attention_cuda: ") +
                                 cudaGetErrorString(attr_err));
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(d_model));
    dim3 grid(static_cast<unsigned int>(seq_len));
    dim3 threads(kThreads);

    block_sparse_attention_kernel<<<grid, threads, smem_bytes, stream>>>(
        static_cast<const float*>(Q.raw_data_gpu()),
        static_cast<const float*>(K.raw_data_gpu()),
        static_cast<const float*>(V.raw_data_gpu()),
        static_cast<float*>(output.raw_data_gpu()),
        d_block_mask, static_cast<int>(seq_len), static_cast<int>(d_model),
        static_cast<int>(block_size), static_cast<int>(num_blocks), scale);

    const cudaError_t err = cudaGetLastError();
    cudaFree(d_block_mask);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("block_sparse_attention_cuda: ") +
                                 cudaGetErrorString(err));
    }

    output.mark_dirty(Tensor::DataLocation::GPU);
    output.syncToHost();

    return output;
}

}  // namespace attention
}  // namespace serius

#endif  // SERIUS_USE_CUDA
