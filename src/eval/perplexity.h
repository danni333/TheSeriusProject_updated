#pragma once

#include <cmath>
#include <vector>

namespace serius {
namespace eval {

/// Perplexity from per-token log-probabilities:
///   PPL = exp( - mean(log_probs) )
float perplexity_from_logprobs(const std::vector<float> &log_probs);

/// Perplexity of a uniform distribution over vocab_size (i.e. log p = -log(vocab_size)).
inline float uniform_perplexity(size_t vocab_size)
{
    return vocab_size > 0 ? static_cast<float>(vocab_size) : 1.0f;
}

}  // namespace eval
}  // namespace serius
