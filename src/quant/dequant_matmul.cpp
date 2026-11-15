#include "dequant_matmul.h"

#include "int4.h"
#include "quant_matmul.h"

namespace serius {
namespace quant {

void dequant_matmul(const uint8_t *W_q4, const float *scales, const uint8_t *zeros, const float *X, float *Y, int M,
                    int N, int K, int block_size)
{
    int blocks_per_row = (K + block_size - 1) / block_size;

    auto decode = [&](int j, int k) -> float {
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
        return (static_cast<float>(q) - static_cast<float>(z)) * s;
    };

    dequant_matmul_t(X, Y, M, N, K, decode);
}

}  // namespace quant
}  // namespace serius
