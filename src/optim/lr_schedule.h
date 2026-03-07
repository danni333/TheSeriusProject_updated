#pragma once

#include <string>

namespace serius {
namespace optim {

class WSDScheduler
{
   public:
    WSDScheduler(float max_lr, int warmup_steps, int stable_steps, int decay_steps);

    /// Get current LR for the given step
    float get_lr(int step) const;

   private:
    float max_lr_;
    int warmup_steps_;
    int stable_steps_;
    int decay_steps_;
};

}  // namespace optim
}  // namespace serius
