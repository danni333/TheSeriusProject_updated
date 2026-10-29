#pragma once

#include "../tensor/tensor.h"

#include <vector>

namespace serius {
namespace ssm {

/// Mamba-style selective SSM block (CPU, sequential scan).
class MambaBlock
{
   public:
    MambaBlock(size_t d_model, size_t d_state, size_t d_conv = 4);
    core::Tensor forward(const core::Tensor &x) const;

   private:
    size_t d_model_;
    size_t d_state_;
    size_t d_conv_;

    // Parameters (stored row-major)
    core::Tensor A_log_;   // [d_state]
    core::Tensor D_;       // [d_model]
    core::Tensor in_proj_; // [2*d_model + d_state*2 + 1, d_model] (simplified: not used)
};

}  // namespace ssm
}  // namespace serius