#pragma once

#include <map>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace interpret {

using core::Tensor;

/// Hooks into intermediate activations to understand which neurons fire for specific inputs.
/// Also supports MoE expert routing analysis.
class ActivationProbe
{
   public:
    ActivationProbe() = default;

    /// Hook an activation output under a given layer name
    void hook(const std::string &layer_name, const Tensor &output);

    /// Get all activations recorded for a layer
    const std::vector<Tensor> &get(const std::string &layer_name) const;

    /// Find top-N most active neurons for a given layer and token position
    /// Returns pairs of (neuron_index, activation_value), sorted descending
    std::vector<std::pair<int, float>> top_neurons(const std::string &layer_name,
                                                   int token_pos, int top_n = 10) const;

    /// Log MoE expert routing for a specific layer and token
    void log_expert_routing(int layer, int token_pos, const std::vector<int> &experts,
                            const std::vector<float> &weights);

    /// Get expert routing log
    struct RoutingEntry
    {
        int layer;
        int token_pos;
        std::vector<int> experts;
        std::vector<float> weights;
    };
    const std::vector<RoutingEntry> &routing_log() const { return routing_log_; }

    /// Clear all hooks and logs
    void clear();

   private:
    std::map<std::string, std::vector<Tensor>> activations_;
    std::vector<RoutingEntry> routing_log_;
};

}  // namespace interpret
}  // namespace serius