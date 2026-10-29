#include "pruner.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace serius {
namespace automl {

Pruner::Pruner(float threshold) : threshold_(threshold)
{
}

PruneResult Pruner::prune(const core::Tensor &W, const std::vector<float> &activation_stats)
{
    PruneResult result;

    if (W.ndim() != 2) {
        result.kept_ratio = 1.0f;
        result.estimated_speedup = 1.0f;
        return result;
    }

    size_t rows = W.shape()[0];
    if (activation_stats.size() != rows) {
        result.kept_ratio = 1.0f;
        result.estimated_speedup = 1.0f;
        return result;
    }

    // Count rows that survive the threshold.
    std::vector<size_t> keep;
    keep.reserve(rows);
    for (size_t i = 0; i < rows; ++i) {
        if (std::fabs(activation_stats[i]) >= threshold_) {
            keep.push_back(i);
        }
    }

    if (keep.empty()) {
        keep.push_back(0);  // Always keep at least one row.
    }

    result.kept_ratio = static_cast<float>(keep.size()) / static_cast<float>(rows);

    // Estimated speedup: FLOPs are proportional to rows kept.
    // A 50% row reduction yields ~2x speedup on the matmul.
    if (result.kept_ratio > 0.0f) {
        result.estimated_speedup = 1.0f / result.kept_ratio;
    } else {
        result.estimated_speedup = 1.0f;
    }

    return result;
}

core::Tensor Pruner::prune_weights(const core::Tensor &W, const std::vector<float> &activation_stats)
{
    if (W.ndim() != 2 || activation_stats.size() != W.shape()[0]) {
        return W.copy();
    }

    size_t rows = W.shape()[0];
    size_t cols = W.shape()[1];

    std::vector<size_t> keep;
    keep.reserve(rows);
    for (size_t i = 0; i < rows; ++i) {
        if (std::fabs(activation_stats[i]) >= threshold_) {
            keep.push_back(i);
        }
    }

    if (keep.empty()) {
        keep.push_back(0);
    }

    core::Tensor pruned({keep.size(), cols});
    for (size_t i = 0; i < keep.size(); ++i) {
        for (size_t j = 0; j < cols; ++j) {
            pruned.at(i, j) = W.at(keep[i], j);
        }
    }

    return pruned;
}

}  // namespace automl
}  // namespace serius