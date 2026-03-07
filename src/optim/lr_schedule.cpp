#include "lr_schedule.h"

#include <algorithm>


namespace serius {
namespace optim {

WSDScheduler::WSDScheduler(float max_lr, int warmup_steps, int stable_steps, int decay_steps)
    : max_lr_(max_lr), warmup_steps_(warmup_steps), stable_steps_(stable_steps), decay_steps_(decay_steps)
{
}

float WSDScheduler::get_lr(int step) const
{
    if (step < warmup_steps_) {
        // Linearity warmup
        return max_lr_ * (static_cast<float>(step) / static_cast<float>(warmup_steps_));
    } else if (step < warmup_steps_ + stable_steps_) {
        // Stable phase
        return max_lr_;
    } else if (step < warmup_steps_ + stable_steps_ + decay_steps_) {
        // Decay phase (linear decay to 0.1 * max_lr)
        int current_decay_step = step - (warmup_steps_ + stable_steps_);
        float decay_progress = static_cast<float>(current_decay_step) / static_cast<float>(decay_steps_);
        return max_lr_ * (1.0f - 0.9f * decay_progress);
    } else {
        // Minimum LR
        return max_lr_ * 0.1f;
    }
}

}  // namespace optim
}  // namespace serius
