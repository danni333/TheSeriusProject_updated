#pragma once

#include <cstddef>
#include <vector>

namespace serius {
namespace neuro {

/// Converts continuous analog values into discrete spike trains, and vice-versa.
class SpikeEncoder
{
   public:
    /// Rate encoding: encodes a value in [0, 1] into a spike train of duration num_steps.
    /// Higher values yield higher firing frequencies.
    /// @param value       Continuous value clamped to [0, 1]
    /// @param num_steps   Length of the time window
    /// @param deterministic  If true, generates uniform deterministic spikes; otherwise pseudo-probabilistic
    static std::vector<bool> rate_encode(float value, size_t num_steps, bool deterministic = true);

    /// Vectorized rate encoding.
    static std::vector<std::vector<bool>> rate_encode_vector(const std::vector<float> &values,
                                                             size_t num_steps,
                                                             bool deterministic = true);

    /// Latency / temporal encoding: stronger inputs spike earlier.
    /// Earliest spike at step 0 for value=1.0, latest spike at step num_steps-1 for value>0, none for 0.
    static std::vector<bool> latency_encode(float value, size_t num_steps);

    /// Vectorized latency encoding.
    static std::vector<std::vector<bool>> latency_encode_vector(const std::vector<float> &values,
                                                                size_t num_steps);

    /// Decode rate: reconstructs original continuous value as (spike_count / num_steps).
    static float decode_rate(const std::vector<bool> &spike_train);

    /// Decode latency: reconstructs value based on time of first spike.
    static float decode_latency(const std::vector<bool> &spike_train);
};

}  // namespace neuro
}  // namespace serius
