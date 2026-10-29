#include "imagination.h"

#include <limits>

namespace serius {
namespace world {

TrajectoryRollout ImaginationEngine::imagine(const WorldModel &wm,
                                             const core::Tensor &initial_state,
                                             const std::vector<core::Tensor> &action_sequence,
                                             float gamma) const
{
    TrajectoryRollout rollout;
    rollout.steps.reserve(action_sequence.size());

    if (action_sequence.empty()) {
        return rollout;
    }

    core::Tensor current_state = initial_state;
    float current_discount = 1.0f;
    float total_unc = 0.0f;

    for (const auto &action : action_sequence) {
        Prediction pred = wm.predict(current_state, action);
        rollout.cumulative_reward += current_discount * pred.predicted_reward;
        total_unc += pred.uncertainty;

        current_state = pred.next_state;
        rollout.steps.push_back(std::move(pred));
        current_discount *= gamma;
    }

    rollout.mean_uncertainty = total_unc / static_cast<float>(action_sequence.size());
    return rollout;
}

int ImaginationEngine::plan_best_action_sequence(const WorldModel &wm,
                                                 const core::Tensor &initial_state,
                                                 const std::vector<std::vector<core::Tensor>> &candidates,
                                                 float gamma) const
{
    if (candidates.empty()) {
        return -1;
    }

    int best_index = 0;
    float best_return = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < candidates.size(); ++i) {
        TrajectoryRollout rollout = imagine(wm, initial_state, candidates[i], gamma);
        if (rollout.cumulative_reward > best_return) {
            best_return = rollout.cumulative_reward;
            best_index = static_cast<int>(i);
        }
    }

    return best_index;
}

}  // namespace world
}  // namespace serius
