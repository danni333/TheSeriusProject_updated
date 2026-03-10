#pragma once

#include "../tensor/tensor.h"

namespace serius {
namespace init {

/// Xavier (Glorot) Uniform initialization.
/// Suitable for layers with linear or tanh activation.
/// W ~ U(-limit, +limit) where limit = sqrt(6 / (fan_in + fan_out))
void xavier_uniform(core::Tensor &w, size_t fan_in, size_t fan_out, unsigned seed = 42);

/// Xavier (Glorot) Normal initialization.
/// W ~ N(0, std) where std = sqrt(2 / (fan_in + fan_out))
void xavier_normal(core::Tensor &w, size_t fan_in, size_t fan_out, unsigned seed = 42);

/// He (Kaiming) Normal initialization.
/// Suitable for layers with ReLU/GELU activation.
/// W ~ N(0, std) where std = sqrt(2 / fan_in)
void he_normal(core::Tensor &w, size_t fan_in, unsigned seed = 42);

/// He (Kaiming) Uniform initialization.
/// W ~ U(-limit, +limit) where limit = sqrt(6 / fan_in)
void he_uniform(core::Tensor &w, size_t fan_in, unsigned seed = 42);

/// Normal initialization with custom mean and standard deviation.
void normal_init(core::Tensor &w, float mean, float std, unsigned seed = 42);

/// Constant initialization (fill all elements with a value).
void constant_init(core::Tensor &w, float value);

/// Initialize embedding weights: N(0, 0.02)
void embedding_init(core::Tensor &w, unsigned seed = 42);

/// Initialize LayerNorm gamma to 1.0, beta to 0.0
void layernorm_init(core::Tensor &gamma, core::Tensor &beta);

}  // namespace init
}  // namespace serius
