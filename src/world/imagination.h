#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"
#include "world_model.h"

namespace serius {
namespace world {

struct TrajectoryRollout
{
    std::vector<Prediction> steps;
    float cumulative_reward = 0.0f;
    float mean_uncertainty = 0.0f;
};

/// Multi-step latent imagination engine.
/// Simulates future consequences entirely inside the latent space of the World Model.
class ImaginationEngine
{
   public:
    ImaginationEngine() = default;

    /// Imagine a multi-step future trajectory given an initial state and an action sequence.
    /// @param wm               World Model instance
    /// @param initial_state    Starting latent state s_0
    /// @param action_sequence  Sequence of candidate actions [a_0, a_1, ..., a_{T-1}]
    /// @param gamma            Discount factor for cumulative reward
    /// @return TrajectoryRollout containing all predicted intermediate steps and totals
    TrajectoryRollout imagine(const WorldModel &wm,
                              const core::Tensor &initial_state,
                              const std::vector<core::Tensor> &action_sequence,
                              float gamma = 0.99f) const;

    /// Plan the optimal action sequence among candidate sequences using Model Predictive Control.
    /// @return Index of the candidate action sequence with the highest cumulative reward
    int plan_best_action_sequence(const WorldModel &wm,
                                  const core::Tensor &initial_state,
                                  const std::vector<std::vector<core::Tensor>> &candidates,
                                  float gamma = 0.99f) const;
};

}  // namespace world
}  // namespace serius
