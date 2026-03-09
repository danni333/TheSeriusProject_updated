#include "layernorm.h"

#include "../autograd/ops.h"
#include "../hal/device.h"  // Forward declarations and IBackend

namespace serius {
namespace model {

using core::Tensor;
using core::autograd::Node;

LayerNorm::LayerNorm(size_t d_model, float epsilon) : d_model_(d_model), eps_(epsilon)
{
    gamma_ = new Node(Tensor({d_model}, hal::DType::Float32));
    beta_ = new Node(Tensor({d_model}, hal::DType::Float32));

    gamma_->data.fill(1.0F);
    beta_->data.zeros();
    gamma_->name = "layernorm_gamma";
    beta_->name = "layernorm_beta";
}

LayerNorm::~LayerNorm()
{
    delete gamma_;
    delete beta_;
}

Node *LayerNorm::forward(Node *input) const
{
    // Simplified LN for example
    return core::autograd::ops::layernorm(input, gamma_, beta_, eps_);
}

}  // namespace model
}  // namespace serius
