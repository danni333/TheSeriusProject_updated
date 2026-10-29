#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace automl {

/// Result of a pruning operation.
struct PruneResult
{
    float kept_ratio;       // Fraction of rows retained
    float estimated_speedup; // Estimated speedup from reduced FLOPs
};

/// Structured pruner: removes entire rows (neurons) from weight matrices
/// based on activation statistics. This is a weight-only structured pruning
/// that preserves the autograd interface by returning a new weight tensor.
class Pruner
{
   public:
    /// Threshold below which a neuron's average absolute activation is pruned.
    /// Higher threshold = more aggressive pruning.
    explicit Pruner(float threshold = 0.01f);

    /// Prune rows of a weight matrix based on activation statistics.
    /// @param W Weight matrix of shape (rows, cols).
    /// @param activation_stats Vector of length `rows` with mean |activation| per row.
    /// @return PruneResult with kept ratio and estimated speedup.
    PruneResult prune(const core::Tensor &W, const std::vector<float> &activation_stats);

    /// Prune rows of a weight matrix and return the pruned weight tensor.
    core::Tensor prune_weights(const core::Tensor &W, const std::vector<float> &activation_stats);

    float threshold() const { return threshold_; }
    void set_threshold(float t) { threshold_ = t; }

   private:
    float threshold_;
};

}  // namespace automl
}  // namespace serius