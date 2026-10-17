#include "dp_noise.h"

#include <random>

namespace serius {
namespace federated {

float compute_l2_norm(const core::Tensor &t)
{
    float sum_sq = 0.0f;
    const float *data = t.data();
    for (size_t i = 0; i < t.size(); ++i) {
        sum_sq += data[i] * data[i];
    }
    return std::sqrt(sum_sq);
}

core::Tensor add_dp_noise(const core::Tensor &delta, float clip_norm, float sigma)
{
    float norm = compute_l2_norm(delta);

    core::Tensor clipped = delta;
    if (norm > clip_norm && norm > 0.0f) {
        float scale = clip_norm / norm;
        clipped = delta * scale;
    }

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, sigma * clip_norm);

    core::Tensor noise(delta.shape());
    float *noise_data = noise.data();
    for (size_t i = 0; i < noise.size(); ++i) {
        noise_data[i] = dist(gen);
    }

    return clipped + noise;
}

float compute_epsilon(float sigma, int num_steps, float target_delta)
{
    if (sigma <= 0.0f || num_steps <= 0) {
        return 0.0f;
    }
    float ln_term = std::log(1.25f / target_delta);
    float eps_single = std::sqrt(2.0f * ln_term) / sigma;

    // Advanced composition
    float ln_delta = std::log(1.0f / target_delta);
    float term1 = std::sqrt(2.0f * static_cast<float>(num_steps) * ln_delta) * eps_single;
    float term2 =
        static_cast<float>(num_steps) * eps_single * (std::exp(eps_single) - 1.0f);
    return term1 + term2;
}

}  // namespace federated
}  // namespace serius
