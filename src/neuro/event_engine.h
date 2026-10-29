#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"
#include "lif_neuron.h"

namespace serius {
namespace neuro {

struct EventStats
{
    size_t total_neurons = 0;
    size_t active_neurons = 0;
    float activity_ratio = 0.0f;
    float sparsity_ratio = 1.0f;
    size_t operations_performed = 0;
    size_t operations_saved = 0;
};

/// Event-driven execution engine for asynchronous spiking neural computation.
/// Computes updates exclusively for active connections, achieving high sparsity.
class EventEngine
{
   public:
    EventEngine() = default;

    /// Execute an event-driven forward step.
    /// @param active_sources    List of neuron indices that fired spikes in the preceding layer
    /// @param weights           Synaptic weight tensor of shape [num_dest, num_source]
    /// @param neurons           Vector of destination LIF neurons
    /// @param weight_threshold  Weights below this magnitude are skipped
    /// @return Indices of destination neurons that emitted spikes
    std::vector<int> step(const std::vector<int> &active_sources,
                          const core::Tensor &weights,
                          std::vector<LIFNeuron> &neurons,
                          float weight_threshold = 1e-6f);

    const EventStats &last_stats() const { return last_stats_; }
    size_t cumulative_ops_performed() const { return cumulative_ops_performed_; }
    size_t cumulative_ops_saved() const { return cumulative_ops_saved_; }

    void reset_stats();

   private:
    EventStats last_stats_{};
    size_t cumulative_ops_performed_ = 0;
    size_t cumulative_ops_saved_ = 0;
};

}  // namespace neuro
}  // namespace serius
