#include "encoder.h"

#include <algorithm>
#include <cmath>

namespace serius {
namespace neuro {

std::vector<bool> SpikeEncoder::rate_encode(float value, size_t num_steps, bool deterministic)
{
    if (num_steps == 0) return {};
    float clamped = std::max(0.0f, std::min(1.0f, value));
    std::vector<bool> spikes(num_steps, false);

    if (clamped <= 1e-6f) {
        return spikes;
    }

    if (deterministic) {
        float accumulator = 0.0f;
        for (size_t t = 0; t < num_steps; ++t) {
            accumulator += clamped;
            if (accumulator >= 1.0f) {
                spikes[t] = true;
                accumulator -= 1.0f;
            }
        }
    } else {
        // Uniform spacing targeting exactly round(clamped * num_steps) spikes
        size_t target_spikes = static_cast<size_t>(std::round(clamped * static_cast<float>(num_steps)));
        for (size_t k = 0; k < target_spikes; ++k) {
            size_t step = (k * num_steps) / std::max<size_t>(1, target_spikes);
            if (step < num_steps) {
                spikes[step] = true;
            }
        }
    }

    return spikes;
}

std::vector<std::vector<bool>> SpikeEncoder::rate_encode_vector(const std::vector<float> &values,
                                                                size_t num_steps,
                                                                bool deterministic)
{
    std::vector<std::vector<bool>> result;
    result.reserve(values.size());
    for (float v : values) {
        result.push_back(rate_encode(v, num_steps, deterministic));
    }
    return result;
}

std::vector<bool> SpikeEncoder::latency_encode(float value, size_t num_steps)
{
    if (num_steps == 0) return {};
    float clamped = std::max(0.0f, std::min(1.0f, value));
    std::vector<bool> spikes(num_steps, false);

    if (clamped <= 1e-6f) {
        return spikes;
    }

    if (num_steps == 1) {
        spikes[0] = true;
        return spikes;
    }

    // High values spike near step 0, lower values spike near step (num_steps - 1)
    size_t spike_time = static_cast<size_t>(std::floor((1.0f - clamped) * static_cast<float>(num_steps - 1)));
    if (spike_time < num_steps) {
        spikes[spike_time] = true;
    }

    return spikes;
}

std::vector<std::vector<bool>> SpikeEncoder::latency_encode_vector(const std::vector<float> &values,
                                                                   size_t num_steps)
{
    std::vector<std::vector<bool>> result;
    result.reserve(values.size());
    for (float v : values) {
        result.push_back(latency_encode(v, num_steps));
    }
    return result;
}

float SpikeEncoder::decode_rate(const std::vector<bool> &spike_train)
{
    if (spike_train.empty()) return 0.0f;
    size_t spike_count = 0;
    for (bool s : spike_train) {
        if (s) spike_count++;
    }
    return static_cast<float>(spike_count) / static_cast<float>(spike_train.size());
}

float SpikeEncoder::decode_latency(const std::vector<bool> &spike_train)
{
    if (spike_train.empty()) return 0.0f;
    if (spike_train.size() == 1) return spike_train[0] ? 1.0f : 0.0f;

    for (size_t t = 0; t < spike_train.size(); ++t) {
        if (spike_train[t]) {
            return 1.0f - (static_cast<float>(t) / static_cast<float>(spike_train.size() - 1));
        }
    }
    return 0.0f;
}

}  // namespace neuro
}  // namespace serius
