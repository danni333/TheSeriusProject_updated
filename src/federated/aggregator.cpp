#include "aggregator.h"

#include <numeric>

namespace serius {
namespace federated {

void FederatedAggregator::add_update(size_t user_id, const core::Tensor &delta_noisy)
{
    updates_[user_id] = delta_noisy;
}

core::Tensor FederatedAggregator::aggregate()
{
    if (updates_.empty()) {
        return core::Tensor({});
    }

    auto it = updates_.begin();
    core::Tensor sum = it->second;
    size_t count = 1;
    ++it;
    for (; it != updates_.end(); ++it) {
        sum = sum + it->second;
        ++count;
    }

    return sum * (1.0f / static_cast<float>(count));
}

void FederatedAggregator::reset()
{
    updates_.clear();
}

}  // namespace federated
}  // namespace serius
