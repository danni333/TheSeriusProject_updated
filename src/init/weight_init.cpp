#include "weight_init.h"

#include <cmath>
#include <random>

namespace serius {
namespace init {

void xavier_uniform(core::Tensor &w, size_t fan_in, size_t fan_out, unsigned seed)
{
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-limit, limit);
    float *data = w.data();
    for (size_t i = 0; i < w.size(); i++) {
        data[i] = dist(gen);
    }
}

void xavier_normal(core::Tensor &w, size_t fan_in, size_t fan_out, unsigned seed)
{
    float std_val = std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, std_val);
    float *data = w.data();
    for (size_t i = 0; i < w.size(); i++) {
        data[i] = dist(gen);
    }
}

void he_normal(core::Tensor &w, size_t fan_in, unsigned seed)
{
    float std_val = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, std_val);
    float *data = w.data();
    for (size_t i = 0; i < w.size(); i++) {
        data[i] = dist(gen);
    }
}

void he_uniform(core::Tensor &w, size_t fan_in, unsigned seed)
{
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-limit, limit);
    float *data = w.data();
    for (size_t i = 0; i < w.size(); i++) {
        data[i] = dist(gen);
    }
}

void normal_init(core::Tensor &w, float mean, float std, unsigned seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(mean, std);
    float *data = w.data();
    for (size_t i = 0; i < w.size(); i++) {
        data[i] = dist(gen);
    }
}

void constant_init(core::Tensor &w, float value)
{
    w.fill(value);
}

void embedding_init(core::Tensor &w, unsigned seed)
{
    normal_init(w, 0.0f, 0.02f, seed);
}

void layernorm_init(core::Tensor &gamma, core::Tensor &beta)
{
    gamma.fill(1.0f);
    beta.fill(0.0f);
}

}  // namespace init
}  // namespace serius
