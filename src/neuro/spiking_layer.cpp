#include "spiking_layer.h"

#include <algorithm>

namespace serius {
namespace neuro {

SpikingLinear::SpikingLinear(size_t in_features, size_t out_features,
                             float threshold, float decay,
                             float reset_value, int refractory_period)
    : in_features_(in_features),
      out_features_(out_features),
      weights_({out_features, in_features}),
      neurons_(out_features, LIFNeuron(threshold, decay, reset_value, refractory_period))
{
    weights_.zeros();
}

std::vector<bool> SpikingLinear::forward(const std::vector<bool> &input_spikes)
{
    std::vector<bool> outputs(out_features_, false);

    for (size_t i = 0; i < out_features_; ++i) {
        float current = 0.0f;
        size_t limit = std::min(input_spikes.size(), in_features_);
        for (size_t j = 0; j < limit; ++j) {
            if (input_spikes[j]) {
                current += weights_[i * in_features_ + j];
            }
        }
        outputs[i] = neurons_[i].step(current);
    }

    return outputs;
}

std::vector<bool> SpikingLinear::forward_currents(const std::vector<float> &currents)
{
    std::vector<bool> outputs(out_features_, false);
    size_t limit = std::min(currents.size(), out_features_);
    for (size_t i = 0; i < limit; ++i) {
        outputs[i] = neurons_[i].step(currents[i]);
    }
    return outputs;
}

void SpikingLinear::apply_stdp(const std::vector<bool> &pre_spikes,
                               const std::vector<bool> &post_spikes,
                               float lr_ltp, float lr_ltd)
{
    size_t in_limit = std::min(pre_spikes.size(), in_features_);
    size_t out_limit = std::min(post_spikes.size(), out_features_);

    for (size_t i = 0; i < out_limit; ++i) {
        if (post_spikes[i]) {
            for (size_t j = 0; j < in_limit; ++j) {
                size_t idx = i * in_features_ + j;
                if (pre_spikes[j]) {
                    // Pre fired before/with post: Long-Term Potentiation (LTP)
                    weights_[idx] += lr_ltp;
                } else {
                    // Post fired without pre: Long-Term Depression (LTD)
                    weights_[idx] -= lr_ltd;
                }
            }
        }
    }
}

void SpikingLinear::reset()
{
    for (auto &n : neurons_) {
        n.reset();
    }
}

}  // namespace neuro
}  // namespace serius
