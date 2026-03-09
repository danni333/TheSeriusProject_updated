#include "layernorm.h"

#include "../../hal/backend.h"  // Added for hal::get_backend
#include "../autograd/ops.h"

namespace serius {
namespace model {

LayerNorm::LayerNorm(size_t d_model, float eps) : d_model_(d_model), eps_(eps)
{
    hal::IBackend *cpu = hal::get_backend(hal::DeviceType::CPU);
    gamma_ = new core::autograd::Node(core::Tensor({d_model}, hal::DType::Float32));
    beta_ = new core::autograd::Node(core::Tensor({d_model}, hal::DType::Float32));

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

core::autograd::Node *LayerNorm::forward(core::autograd::Node *input) const
{
    // Simplified LN for example
    return autograd::ops::layernorm(input, gamma_, beta_, eps_);
}

}  // namespace model
}  // namespace serius
