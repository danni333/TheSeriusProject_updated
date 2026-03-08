#include "dequant_matmul.h"

#include "int4.h"

namespace serius {
namespace quant {

void dequant_matmul(const uint8_t *W_q4, const float *scales, const uint8_t *zeros, const float *X, float *Y, int M,
                    int N, int K, int block_size)
{
    int blocks_per_row = (K + block_size - 1) / block_size;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                int block_idx = k / block_size;
                float s = scales[j * blocks_per_row + block_idx];
                uint8_t z = zeros[j * blocks_per_row + block_idx];

                // Unpack the 4-bit weight from the packed array
                int packed_idx = (j * K + k) / 2;
                uint8_t packed = W_q4[packed_idx];
                uint8_t q;
                if ((k % 2) == 0) {
                    q = packed & 0x0F;
                } else {
                    q = (packed >> 4) & 0x0F;
                }

                // Dequantize in register
                float w = (static_cast<float>(q) - static_cast<float>(z)) * s;
                sum += X[i * K + k] * w;
            }

            Y[i * N + j] = sum;
        }
    }
}

}  // namespace quant
}  // namespace serius
