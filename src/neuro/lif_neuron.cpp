#include "lif_neuron.h"

namespace serius {
namespace neuro {

LIFNeuron::LIFNeuron(float threshold, float decay, float reset_value, int refractory_period)
    : membrane_potential_(reset_value),
      threshold_(threshold),
      decay_(decay),
      reset_value_(reset_value),
      refractory_period_(refractory_period),
      refractory_timer_(0)
{
}

bool LIFNeuron::step(float input_current)
{
    if (refractory_timer_ > 0) {
        refractory_timer_--;
        membrane_potential_ = reset_value_;
        return false;
    }

    // Leaky integration
    membrane_potential_ = decay_ * membrane_potential_ + input_current;

    // Spike threshold check
    if (membrane_potential_ >= threshold_) {
        membrane_potential_ = reset_value_;
        refractory_timer_ = refractory_period_;
        return true;
    }

    return false;
}

void LIFNeuron::reset()
{
    membrane_potential_ = reset_value_;
    refractory_timer_ = 0;
}

}  // namespace neuro
}  // namespace serius
