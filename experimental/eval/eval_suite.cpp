#include "eval_suite.h"

namespace serius {
namespace eval {

EvalResult eval_multiple_choice(const std::string &benchmark_name,
                                const std::vector<int> &predictions,
                                const std::vector<int> &ground_truth)
{
    EvalResult result;
    result.benchmark = benchmark_name;
    result.total = static_cast<int>(std::min(predictions.size(), ground_truth.size()));
    for (int i = 0; i < result.total; ++i) {
        if (predictions[i] == ground_truth[i]) {
            ++result.correct;
        }
    }
    return result;
}

}  // namespace eval
}  // namespace serius
