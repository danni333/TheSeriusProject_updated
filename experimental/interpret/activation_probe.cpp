#include "activation_probe.h"

#include <algorithm>
#include <cmath>

namespace serius {
namespace interpret {

void ActivationProbe::hook(const std::string &layer_name, const Tensor &output)
{
    activations_[layer_name].push_back(output.copy());
}

const std::vector<Tensor> &ActivationProbe::get(const std::string &layer_name) const
{
    auto it = activations_.find(layer_name);
    if (it != activations_.end()) {
        return it->second;
    }
    static const std::vector<Tensor> empty;
    return empty;
}

std::vector<std::pair<int, float>> ActivationProbe::top_neurons(const std::string &layer_name,
                                                                 int token_pos, int top_n) const
{
    std::vector<std::pair<int, float>> result;
    auto it = activations_.find(layer_name);
    if (it == activations_.end() || it->second.empty()) {
        return result;
    }

    const Tensor &act = it->second[0];
    size_t d = act.shape().back();
    result.reserve(d);

    for (size_t i = 0; i < d; ++i) {
        float val = 0.0f;
        if (act.ndim() == 2) {
            val = std::abs(act.item(token_pos, i));
        } else {
            val = std::abs(act.item(i));
        }
        result.push_back({static_cast<int>(i), val});
    }

    std::sort(result.begin(), result.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    if (static_cast<int>(result.size()) > top_n) {
        result.resize(top_n);
    }
    return result;
}

void ActivationProbe::log_expert_routing(int layer, int token_pos, const std::vector<int> &experts,
                                         const std::vector<float> &weights)
{
    RoutingEntry entry;
    entry.layer = layer;
    entry.token_pos = token_pos;
    entry.experts = experts;
    entry.weights = weights;
    routing_log_.push_back(entry);
}

void ActivationProbe::clear()
{
    activations_.clear();
    routing_log_.clear();
}

}  // namespace interpret
}  // namespace serius