#include "perplexity.h"

namespace serius {
namespace eval {

float perplexity_from_logprobs(const std::vector<float> &log_probs)
{
    if (log_probs.empty()) {
        return 1.0f;
    }
    float sum = 0.0f;
    for (float lp : log_probs) {
        sum += lp;
    }
    const float avg = sum / static_cast<float>(log_probs.size());
    return std::exp(-avg);
}

}  // namespace eval
}  // namespace serius
