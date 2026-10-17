#pragma once

#include <map>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace federated {

/// Server-side secure aggregation of noisy weight deltas from multiple users.
class FederatedAggregator
{
   public:
    FederatedAggregator() = default;

    /// Add a noisy delta update from a user.
    void add_update(size_t user_id, const core::Tensor &delta_noisy);

    /// Average all collected updates and return the global delta.
    /// Returns a zero tensor with the same shape as the first update if none collected.
    core::Tensor aggregate();

    /// Clear all collected updates.
    void reset();

    size_t num_updates() const { return updates_.size(); }

   private:
    std::map<size_t, core::Tensor> updates_;
};

}  // namespace federated
}  // namespace serius
