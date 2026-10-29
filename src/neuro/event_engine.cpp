#include "event_engine.h"

#include <cmath>

namespace serius {
namespace neuro {

std::vector<int> EventEngine::step(const std::vector<int> &active_sources,
                                   const core::Tensor &weights,
                                   std::vector<LIFNeuron> &neurons,
                                   float weight_threshold)
{
    const size_t num_dst = neurons.size();
    if (num_dst == 0 || weights.shape().empty()) {
        return {};
    }

    const size_t num_src = weights.shape().size() > 1 ? weights.shape()[1] : weights.size() / num_dst;
    const size_t max_total_ops = num_dst * num_src;

    std::vector<float> incoming_current(num_dst, 0.0f);
    size_t ops_performed = 0;

    for (int src : active_sources) {
        if (src < 0 || static_cast<size_t>(src) >= num_src) {
            continue;
        }

        for (size_t dst = 0; dst < num_dst; ++dst) {
            float w = weights[dst * num_src + static_cast<size_t>(src)];
            if (std::abs(w) < weight_threshold) {
                continue;
            }
            incoming_current[dst] += w;
            ops_performed++;
        }
    }

    std::vector<int> next_active;
    for (size_t dst = 0; dst < num_dst; ++dst) {
        if (neurons[dst].step(incoming_current[dst])) {
            next_active.push_back(static_cast<int>(dst));
        }
    }

    size_t ops_saved = (max_total_ops >= ops_performed) ? (max_total_ops - ops_performed) : 0;

    last_stats_.total_neurons = num_dst;
    last_stats_.active_neurons = next_active.size();
    last_stats_.activity_ratio = num_dst > 0 ? (static_cast<float>(next_active.size()) / static_cast<float>(num_dst)) : 0.0f;
    last_stats_.sparsity_ratio = 1.0f - last_stats_.activity_ratio;
    last_stats_.operations_performed = ops_performed;
    last_stats_.operations_saved = ops_saved;

    cumulative_ops_performed_ += ops_performed;
    cumulative_ops_saved_ += ops_saved;

    return next_active;
}

void EventEngine::reset_stats()
{
    last_stats_ = EventStats{};
    cumulative_ops_performed_ = 0;
    cumulative_ops_saved_ = 0;
}

}  // namespace neuro
}  // namespace serius
