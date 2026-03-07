#pragma once

#include "../autograd/node.h"
#include "../tensor/tensor.h"


namespace serius {
namespace transformer {

class LayerNorm
{
   public:
    explicit LayerNorm(size_t d_model, float eps = 1e-5F);
    ~LayerNorm();

    /// Forward pass: normalizes along the last dimension
    /// Input shape: (seq_len, d_model) or (batch, seq_len, d_model)
    autograd::Node *forward(autograd::Node *input) const;

    // Learnable parameters (managed by the layer)
    autograd::Node *gamma;  // scale, shape: (d_model,)
    autograd::Node *beta;   // shift, shape: (d_model,)

   private:
    size_t d_model_;
    float eps_;
};

}  // namespace transformer
}  // namespace serius
