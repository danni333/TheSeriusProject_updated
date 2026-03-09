#pragma once

#include "../autograd/node.h"
#include "../tensor/tensor.h"

namespace serius {
namespace model {

class LayerNorm
{
   public:
    explicit LayerNorm(size_t d_model, float eps = 1e-5F);
    ~LayerNorm();

    /// Forward pass: normalizes along the last dimension
    /// Input shape: (seq_len, d_model) or (batch, seq_len, d_model)
    core::autograd::Node *forward(core::autograd::Node *input) const;

    // Learnable parameters (managed by the layer)
    core::autograd::Node *gamma_;  // scale, shape: (d_model,)
    core::autograd::Node *beta_;   // shift, shape: (d_model,)
    float eps_;

   private:
    size_t d_model_;
};

}  // namespace model
}  // namespace serius
