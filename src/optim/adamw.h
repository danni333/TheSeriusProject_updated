#pragma once

#include <map>
#include <vector>

#include "../autograd/node.h"

namespace serius {
namespace optim {

struct AdamWConfig
{
    float lr = 3e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
};

struct ParamState
{
    core::Tensor m;  // 1st moment
    core::Tensor v;  // 2nd moment
    int step = 0;

    ParamState() : m({0}), v({0}), step(0) {}
    ParamState(core::Tensor m_in, core::Tensor v_in, int step_in)
        : m(std::move(m_in)), v(std::move(v_in)), step(step_in)
    {
    }
};

class AdamW
{
   public:
    AdamW(const std::vector<core::autograd::Node *> &params, const AdamWConfig &config = AdamWConfig());

    /// Perform a single optimization step
    void step();

    /// Reset gradients of all parameters to zero
    void zero_grad();

    // State management
    const std::map<core::autograd::Node *, ParamState> &state() const { return state_; }
    std::map<core::autograd::Node *, ParamState> &state() { return state_; }
    void set_lr(float lr) { config_.lr = lr; }

   private:
    std::vector<core::autograd::Node *> params_;
    AdamWConfig config_;
    std::map<core::autograd::Node *, ParamState> state_;
};

}  // namespace optim
}  // namespace serius
