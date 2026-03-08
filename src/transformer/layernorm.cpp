#include "layernorm.h"

#include <cmath>
#include <stdexcept>

#include "../autograd/ops.h"

namespace serius {
namespace transformer {

LayerNorm::LayerNorm(size_t d_model, float eps) : d_model_(d_model), eps(eps)
{
    gamma = new autograd::Node(Tensor({d_model}, hal::DType::Float32), true);
    beta = new autograd::Node(Tensor({d_model}, hal::DType::Float32), true);
    gamma->data.fill(1.0F);
    beta->data.fill(0.0F);
    gamma->name = "layernorm_gamma";
    beta->name = "layernorm_beta";
}

LayerNorm::~LayerNorm()
{
    delete gamma;
    delete beta;
}

autograd::Node *LayerNorm::forward(autograd::Node *input) const
{
    return autograd::ops::layernorm(input, gamma, beta, eps);
}

}  // namespace transformer
}  // namespace serius
