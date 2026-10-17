#include "batcher.h"

namespace serius {
namespace server {

ContinuousBatcher::ContinuousBatcher(size_t max_batch_size)
    : max_batch_size_(max_batch_size), slots_(max_batch_size)
{
}

ContinuousBatcher::~ContinuousBatcher() = default;

void ContinuousBatcher::enqueue(int request_id, std::vector<int> tokens,
                                std::function<void(std::string, bool)> cb)
{
    std::lock_guard<std::mutex> lock(mutex_);
    waiting_.push({request_id, std::move(tokens), std::move(cb)});
}

void ContinuousBatcher::iteration(
    std::function<std::vector<int>(const std::vector<std::vector<int>> &)> forward_batch)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Fill idle slots with waiting requests
    for (auto &slot : slots_) {
        if (!slot.active && !waiting_.empty()) {
            Request req = waiting_.front();
            waiting_.pop();
            slot.request_id = req.id;
            slot.tokens = std::move(req.tokens);
            slot.callback = std::move(req.callback);
            slot.active = true;
        }
    }

    // 2. Collect active slots
    std::vector<size_t> active_indices;
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].active) {
            active_indices.push_back(i);
        }
    }

    if (active_indices.empty()) {
        return;
    }

    // 3. Batched forward: one next token per active slot
    std::vector<std::vector<int>> batch_inputs;
    batch_inputs.reserve(active_indices.size());
    for (size_t idx : active_indices) {
        batch_inputs.push_back(slots_[idx].tokens);
    }

    std::vector<int> next_tokens = forward_batch(batch_inputs);

    // 4. Dispatch per slot
    for (size_t i = 0; i < active_indices.size(); ++i) {
        size_t slot_idx = active_indices[i];
        auto &slot = slots_[slot_idx];
        int next_token = next_tokens[i];
        slot.tokens.push_back(next_token);

        std::string text = std::to_string(next_token);
        bool done = (next_token == -1 || slot.tokens.size() >= 16);  // -1 = EOS, max 16 tokens
        slot.callback(text, done);

        if (done) {
            slot.active = false;
            slot.request_id = -1;
            slot.tokens.clear();
            slot.callback = nullptr;
        }
    }
}

size_t ContinuousBatcher::active_slots() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    size_t n = 0;
    for (const auto &slot : slots_) {
        if (slot.active) {
            ++n;
        }
    }
    return n;
}

size_t ContinuousBatcher::pending() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return waiting_.size();
}

}  // namespace server
}  // namespace serius
