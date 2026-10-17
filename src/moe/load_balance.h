#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace model {

float load_balance_loss(const core::Tensor &router_probs,
                        const std::vector<std::vector<size_t>> &assignments,
                        size_t n_experts,
                        float alpha = 0.01f);

float load_balance_loss(const core::Tensor &router_probs,
                        const std::vector<std::vector<int>> &assignments,
                        size_t n_experts,
                        float alpha = 0.01f);

}  // namespace model
}  // namespace serius
