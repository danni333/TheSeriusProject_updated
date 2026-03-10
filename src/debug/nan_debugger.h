#pragma once

#include "../tensor/tensor.h"

#include <string>
#include <vector>

namespace serius {
namespace debug {

/// Pair (name, tensor) for inspection.
using NamedTensor = std::pair<std::string, const core::Tensor *>;

/// Find the first tensor in the list that contains NaN. Returns its name or empty string.
std::string find_first_nan(const std::vector<NamedTensor> &tensors);

/// Check a single tensor for NaN/Inf. Returns true if any found.
bool has_nan_or_inf(const core::Tensor &t);

}  // namespace debug
}  // namespace serius
