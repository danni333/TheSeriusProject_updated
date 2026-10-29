#pragma once

namespace serius {
namespace neuro {

/// Leaky Integrate-and-Fire (LIF) neuron model.
class LIFNeuron
{
   public:
    explicit LIFNeuron(float threshold = 1.0f, float decay = 0.9f,
                       float reset_value = 0.0f, int refractory_period = 0);

    /// Step the neuron with an input current.
    /// Returns true if the neuron emits a spike.
    bool step(float input_current);

    /// Reset neuron state to resting potential.
    void reset();

    float membrane_potential() const { return membrane_potential_; }
    void set_membrane_potential(float v) { membrane_potential_ = v; }

    float threshold() const { return threshold_; }
    void set_threshold(float th) { threshold_ = th; }

    float decay() const { return decay_; }
    void set_decay(float d) { decay_ = d; }

    float reset_value() const { return reset_value_; }
    void set_reset_value(float r) { reset_value_ = r; }

    int refractory_period() const { return refractory_period_; }
    void set_refractory_period(int p) { refractory_period_ = p; }

    bool is_refractory() const { return refractory_timer_ > 0; }

   private:
    float membrane_potential_;
    float threshold_;
    float decay_;
    float reset_value_;
    int refractory_period_;
    int refractory_timer_;
};

}  // namespace neuro
}  // namespace serius
