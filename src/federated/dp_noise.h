#pragma once

#include <cmath>
#include <string>

#include "../tensor/tensor.h"

namespace serius {
namespace federated {

/// L2 norm of a tensor (Flattened Euclidean norm).
float compute_l2_norm(const core::Tensor &t);

/// Apply differential privacy to a weight delta:
/// 1. Clip L2 norm to clip_norm
/// 2. Add Gaussian noise N(0, sigma^2 * clip_norm^2 * I)
core::Tensor add_dp_noise(const core::Tensor &delta, float clip_norm, float sigma);

/// Approximate (epsilon, delta)-DP budget for the Gaussian mechanism
/// over num_steps compositions using advanced composition.
float compute_epsilon(float sigma, int num_steps, float target_delta = 1e-5f);

}  // namespace federated
}  // namespace serius
