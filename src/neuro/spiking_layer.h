#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"
#include "lif_neuron.h"

namespace serius {
namespace neuro {

/// Spiking linear layer with synaptic weights and LIF neurons.
class SpikingLinear
{
   public:
    SpikingLinear(size_t in_features, size_t out_features,
                  float threshold = 1.0f, float decay = 0.9f,
                  float reset_value = 0.0f, int refractory_period = 0);

    /// Forward pass given binary input spikes.
    /// Computes synaptic currents and steps the layer's LIF neurons.
    std::vector<bool> forward(const std::vector<bool> &input_spikes);

    /// Forward pass with continuous analog currents directly feeding the neurons.
    std::vector<bool> forward_currents(const std::vector<float> &currents);

    /// Apply Spike-Timing-Dependent Plasticity (STDP) weight update.
    /// @param pre_spikes   Input spikes at current time step
    /// @param post_spikes  Output spikes at current time step
    /// @param lr_ltp       Learning rate for Long-Term Potentiation (pre and post fired)
    /// @param lr_ltd       Learning rate for Long-Term Depression (post fired without pre)
    void apply_stdp(const std::vector<bool> &pre_spikes,
                    const std::vector<bool> &post_spikes,
                    float lr_ltp = 0.01f, float lr_ltd = 0.005f);

    /// Reset all neurons to resting potential.
    void reset();

    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }

    core::Tensor &weights() { return weights_; }
    const core::Tensor &weights() const { return weights_; }

    std::vector<LIFNeuron> &neurons() { return neurons_; }
    const std::vector<LIFNeuron> &neurons() const { return neurons_; }

   private:
    size_t in_features_;
    size_t out_features_;
    core::Tensor weights_;             // [out_features, in_features]
    std::vector<LIFNeuron> neurons_;    // [out_features]
};

}  // namespace neuro
}  // namespace serius
