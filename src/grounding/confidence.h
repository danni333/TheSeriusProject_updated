#pragma once

#include <cmath>
#include <vector>

namespace serius {
namespace grounding {

/// Confidence from average log-probability: exp(avg_log_prob), in [0, 1].
float confidence_from_logprobs(const std::vector<float> &log_probs);

}  // namespace grounding
}  // namespace serius
