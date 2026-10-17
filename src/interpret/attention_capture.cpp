#include "attention_capture.h"

#include <cmath>
#include <sstream>

namespace serius {
namespace interpret {

void AttentionCapture::record(int layer, int head, const Tensor &weights)
{
    if (!capturing_) return;
    if (captures_.size() <= static_cast<size_t>(layer)) {
        captures_.resize(layer + 1);
    }
    if (captures_[layer].size() <= static_cast<size_t>(head)) {
        captures_[layer].resize(head + 1);
    }
    captures_[layer][head] = weights.copy();
}

Tensor AttentionCapture::get(int layer, int head) const
{
    if (layer < static_cast<int>(captures_.size()) && head < static_cast<int>(captures_[layer].size())) {
        return captures_[layer][head];
    }
    return Tensor();
}

bool AttentionCapture::validate_softmax(int layer, int head, float tolerance) const
{
    Tensor attn = get(layer, head);
    if (attn.size() == 0) return false;

    size_t seq = attn.shape()[0];
    for (size_t i = 0; i < seq; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < seq; ++j) {
            sum += attn[i * seq + j];
        }
        if (std::abs(sum - 1.0f) > tolerance) {
            return false;
        }
    }
    return true;
}

void AttentionCapture::clear()
{
    captures_.clear();
}

}  // namespace interpret
}  // namespace serius