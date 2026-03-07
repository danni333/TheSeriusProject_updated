#define _USE_MATH_DEFINES
#include "positional.h"

#include <cmath>
#include <stdexcept>

namespace serius {
namespace transformer {

Tensor positional_encoding(int max_seq_len, int d_model)
{
    Tensor pe({static_cast<size_t>(max_seq_len), static_cast<size_t>(d_model)});

    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int idx = 0; idx < d_model; idx += 2) {
            float freq = 1.0F / std::pow(10000.0F, static_cast<float>(idx) / static_cast<float>(d_model));
            pe.at(static_cast<size_t>(pos), static_cast<size_t>(idx)) = std::sin(static_cast<float>(pos) * freq);
            if (idx + 1 < d_model) {
                pe.at(static_cast<size_t>(pos), static_cast<size_t>(idx + 1)) =
                    std::cos(static_cast<float>(pos) * freq);
            }
        }
    }

    return pe;
}

void apply_rope(Tensor &q, Tensor &k, size_t d_head)
{
    if (d_head % 2 != 0) {
        throw std::invalid_argument("RoPE requires d_head to be even");
    }

    bool is_batched = (q.ndim() == 3);
    size_t batch_size = is_batched ? q.shape()[0] : 1;
    size_t seq_len = is_batched ? q.shape()[1] : q.shape()[0];
    const size_t q_dmodel = is_batched ? q.shape()[2] : q.shape()[1];
    const size_t k_dmodel = is_batched ? k.shape()[2] : k.shape()[1];

    if (q_dmodel % d_head != 0 || k_dmodel % d_head != 0) {
        throw std::invalid_argument("last dim of q and k must be divisible by d_head");
    }

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t pos = 0; pos < seq_len; ++pos) {
            // Iterate through each head's dimensions
            for (size_t head_offset_q = 0; head_offset_q < q_dmodel; head_offset_q += d_head) {
                for (size_t i = 0; i < d_head; i += 2) {
                    float freq = 1.0F / std::pow(10000.0F, static_cast<float>(i) / static_cast<float>(d_head));
                    float theta = static_cast<float>(pos) * freq;
                    float cos_theta = std::cos(theta);
                    float sin_theta = std::sin(theta);

                    // Q pairs
                    float q0 = is_batched ? q.at(b, pos, head_offset_q + i) : q.at(pos, head_offset_q + i);
                    float q1 = is_batched ? q.at(b, pos, head_offset_q + i + 1) : q.at(pos, head_offset_q + i + 1);

                    if (is_batched) {
                        q.at(b, pos, head_offset_q + i) = q0 * cos_theta - q1 * sin_theta;
                        q.at(b, pos, head_offset_q + i + 1) = q0 * sin_theta + q1 * cos_theta;
                    } else {
                        q.at(pos, head_offset_q + i) = q0 * cos_theta - q1 * sin_theta;
                        q.at(pos, head_offset_q + i + 1) = q0 * sin_theta + q1 * cos_theta;
                    }
                }
            }

            for (size_t head_offset_k = 0; head_offset_k < k_dmodel; head_offset_k += d_head) {
                for (size_t i = 0; i < d_head; i += 2) {
                    float freq = 1.0F / std::pow(10000.0F, static_cast<float>(i) / static_cast<float>(d_head));
                    float theta = static_cast<float>(pos) * freq;
                    float cos_theta = std::cos(theta);
                    float sin_theta = std::sin(theta);

                    // K pairs
                    float k0 = is_batched ? k.at(b, pos, head_offset_k + i) : k.at(pos, head_offset_k + i);
                    float k1 = is_batched ? k.at(b, pos, head_offset_k + i + 1) : k.at(pos, head_offset_k + i + 1);

                    if (is_batched) {
                        k.at(b, pos, head_offset_k + i) = k0 * cos_theta - k1 * sin_theta;
                        k.at(b, pos, head_offset_k + i + 1) = k0 * sin_theta + k1 * cos_theta;
                    } else {
                        k.at(pos, head_offset_k + i) = k0 * cos_theta - k1 * sin_theta;
                        k.at(pos, head_offset_k + i + 1) = k0 * sin_theta + k1 * cos_theta;
                    }
                }
            }
        }
    }
}

}  // namespace transformer
}  // namespace serius
