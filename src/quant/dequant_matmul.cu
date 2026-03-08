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
__global__ void dequant_matmul_kernel(const uint8_t *__restrict__ W_q4, const float *__restrict__ scales,
                                      const uint8_t *__restrict__ zeros, const float *__restrict__ X,
                                      float *__restrict__ Y, int M, int N, int K, int block_size)
{
    // Each block handles one (i, j) output element
    int j = blockIdx.x;  // output column (weight row)
    int i = blockIdx.y;  // output row

    if (i >= M || j >= N) return;

    int blocks_per_row = (K + block_size - 1) / block_size;
    int tid = threadIdx.x;

    float local_sum = 0.0f;

    // Stride through K dimension
    for (int k = tid; k < K; k += blockDim.x) {
        int block_idx = k / block_size;
        float s = scales[j * blocks_per_row + block_idx];
        uint8_t z = zeros[j * blocks_per_row + block_idx];

        int packed_idx = (j * K + k) / 2;
        uint8_t packed = W_q4[packed_idx];
        uint8_t q;
        if ((k & 1) == 0) {
            q = packed & 0x0F;
        } else {
            q = (packed >> 4) & 0x0F;
        }

        float w = (static_cast<float>(q) - static_cast<float>(z)) * s;
        local_sum += X[i * K + k] * w;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // First thread in each warp writes to shared memory
    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int wid = tid / 32;

    if (lane == 0) {
        warp_sums[wid] = local_sum;
    }
    __syncthreads();

    // First warp reduces the warp sums
    if (wid == 0) {
        float val = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (tid == 0) {
            Y[i * N + j] = val;
        }
    }
}

void dequant_matmul_cuda(const uint8_t *W_q4, const float *scales, const uint8_t *zeros, const float *X, float *Y,
                         int M, int N, int K, int block_size)
{
    int threads = 128;
    if (K > 256) threads = 256;

    dim3 grid(N, M);
    dim3 block(threads);

    dequant_matmul_kernel<<<grid, block>>>(W_q4, scales, zeros, X, Y, M, N, K, block_size);

    cudaDeviceSynchronize();
}

}  // namespace quant
}  // namespace serius
