#pragma once

#include <cstddef>

#include "../tensor/tensor.h"
#include "ltc.h"

namespace serius {
namespace liquid {

/// Adaptive step controller for liquid neural systems.
/// Allocates computational depth / integration steps dynamically based on input complexity.
class AdaptiveStepController
{
   public:
    explicit AdaptiveStepController(float tolerance = 1e-3f, int min_steps = 2, int max_steps = 30);

    /// Estimate numerical complexity score for an input tensor.
    /// Returns a scalar in [0, 1] where 0 is minimal complexity and 1 is maximal.
    static float estimate_complexity(const core::Tensor &x);

    /// Compute the required number of solver steps based on input complexity.
    int compute_steps(const core::Tensor &x) const;

    /// Iterate LTC network dynamics until state convergence or max_steps reached.
    /// @param ltc        LTC network instance
    /// @param x          Input tensor
    /// @param h          Hidden state (updated in-place to final state)
    /// @param tolerance  Convergence threshold: ||h_new - h|| < tolerance
    /// @return Number of steps executed
    int solve_until_convergence(const LTCNetwork &ltc, const core::Tensor &x,
                                core::Tensor &h, float tolerance = -1.0f) const;

    float tolerance() const { return tolerance_; }
    void set_tolerance(float tol) { tolerance_ = tol; }

    int min_steps() const { return min_steps_; }
    int max_steps() const { return max_steps_; }

   private:
    float tolerance_;
    int min_steps_;
    int max_steps_;
};

}  // namespace liquid
}  // namespace serius
