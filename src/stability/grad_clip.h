#pragma once

#include <vector>

#include "../autograd/node.h"

namespace serius {
namespace stability {

/// Clip the global L2 norm of all gradients.
/// Returns the original (unclipped) total norm.
/// If total_norm > max_norm, all gradients are scaled down proportionally.
float clip_grad_norm(const std::vector<core::autograd::Node *> &params, float max_norm);

/// Clip individual gradient values to [-clip_value, +clip_value].
void clip_grad_value(std::vector<core::autograd::Node *> &params, float clip_value);

/// Check if any gradient contains NaN or Inf values.
/// Returns true if NaN/Inf is found.
bool has_nan_or_inf(const std::vector<core::autograd::Node *> &params);

}  // namespace stability
}  // namespace serius
