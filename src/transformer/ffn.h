#pragma once

#include "../autograd/node.h"
#include "../tensor/tensor.h"

namespace serius {
namespace model {

using core::Tensor;
using core::autograd::Node;

class FFN
{
   public:
    FFN(size_t d_model, size_t d_ff);
    ~FFN();

    /// Forward pass: GELU(x·W1 + b1)·W2 + b2
    /// Input: (seq_len, d_model)
    /// Returns: (seq_len, d_model)
    Node *forward(Node *input) const;

    // Parameters
    Node *w1;  // (d_model, d_ff)
    Node *b1;  // (d_ff,)
    Node *w2;  // (d_ff, d_model)
    Node *b2;  // (d_model,)

   private:
    size_t d_model_;
    size_t d_ff_;
};

}  // namespace model
}  // namespace serius
