#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "dequant_matmul.h"

namespace serius {
namespace quant {

/// CUDA kernel: one thread-block computes one output element Y[i][j].
/// Each thread accumulates a portion of the inner product, then
/// warp-level reduction is used to produce the final sum.
///
/// W_q4 is laid out as [N, K/2] packed bytes.
/// scales/zeros are laid out as [N, blocks_per_row].
// Optimized tiled kernel for dequantized matmul (X @ W^T)
// X is [M, K], W_q4 is [N, K/2] packed
// Each thread block computes a TILE_M x TILE_N block of output Y.
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void dequant_matmul_tiled_kernel(
    const uint8_t *__restrict__ W_q4, 
    const float *__restrict__ scales,
    const uint8_t *__restrict__ zeros, 
    const float *__restrict__ X,
    float *__restrict__ Y, 
    int M, int N, int K, int block_size)
{
    __shared__ float s_X[TILE_M][TILE_K];
    __shared__ uint8_t s_W[TILE_N][TILE_K / 2];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;
    
    float accum = 0.0f;
    int blocks_per_row = (K + block_size - 1) / block_size;
    
    for (int k_off = 0; k_off < K; k_off += TILE_K) {
        // Load X into shared memory
        if (row < M && (k_off + tx) < K) {
            s_X[ty][tx] = X[row * K + k_off + tx];
        } else {
            s_X[ty][tx] = 0.0f;
        }
        
        // Load W_q4 into shared memory (packed bytes)
        // tx is 0..15, ty is 0..15
        // We need to load TILE_N * (TILE_K/2) bytes = 16 * 8 = 128 bytes.
        // Total threads = 256. 
        int thread_id = ty * TILE_N + tx;
        if (thread_id < TILE_N * (TILE_K / 2)) {
            int w_col = thread_id / (TILE_K / 2);
            int w_k = thread_id % (TILE_K / 2);
            int global_col = blockIdx.x * TILE_N + w_col;
            if (global_col < N && (k_off + w_k * 2) < K) {
                s_W[w_col][w_k] = W_q4[(global_col * K + k_off + w_k * 2) / 2];
            } else {
                s_W[w_col][w_k] = 0;
            }
        }
        
        __syncthreads();
        
        // Compute partial dot product
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                int global_k = k_off + k;
                if (global_k >= K) break;

                int b_idx = global_k / block_size;
                float s = scales[col * blocks_per_row + b_idx];
                uint8_t z = zeros[col * blocks_per_row + b_idx];
                
                uint8_t packed = s_W[tx][k/2];
                uint8_t q = (k & 1) ? (packed >> 4) : (packed & 0x0F);
                
                float w = (static_cast<float>(q) - static_cast<float>(z)) * s;
                accum += s_X[ty][k] * w;
            }
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        Y[row * N + col] = accum;
    }
}

void dequant_matmul_cuda(const uint8_t *W_q4, const float *scales, const uint8_t *zeros, const float *X, float *Y,
                         int M, int N, int K, int block_size)
{
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    dim3 block(16, 16);

    dequant_matmul_tiled_kernel<16, 16, 16><<<grid, block>>>(W_q4, scales, zeros, X, Y, M, N, K, block_size);

    cudaDeviceSynchronize();
}

}  // namespace quant
}  // namespace serius
