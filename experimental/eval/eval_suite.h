#pragma once

#include <string>
#include <vector>

namespace serius {
namespace eval {

struct EvalResult
{
    std::string benchmark;
    int correct = 0;
    int total = 0;

    float accuracy() const
    {
        return total > 0 ? static_cast<float>(correct) / static_cast<float>(total) : 0.0f;
    }
};

/// Multiple-choice evaluation given per-question predictions and ground truth.
EvalResult eval_multiple_choice(const std::string &benchmark_name,
                                const std::vector<int> &predictions,
                                const std::vector<int> &ground_truth);

}  // namespace eval
}  // namespace serius
