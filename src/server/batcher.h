#pragma once

#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace serius {
namespace server {

struct Request
{
    int id;
    std::vector<int> tokens;
    std::function<void(std::string, bool)> callback;
};

struct Slot
{
    int request_id = -1;
    std::vector<int> tokens;
    std::function<void(std::string, bool)> callback;
    bool active = false;
};

/// Continuous batching engine: fills idle slots with waiting requests,
/// runs a batched forward, samples and dispatches tokens per slot.
/// Frees and reuses slots on EOS / max length.
class ContinuousBatcher
{
   public:
    explicit ContinuousBatcher(size_t max_batch_size);
    ~ContinuousBatcher();

    ContinuousBatcher(const ContinuousBatcher &) = delete;
    ContinuousBatcher &operator=(const ContinuousBatcher &) = delete;

    /// Enqueue a new request.
    void enqueue(int request_id, std::vector<int> tokens, std::function<void(std::string, bool)> cb);

    /// Run one batching iteration.
    /// forward_batch: takes a vector of token sequences (one per active slot)
    /// and returns the next token for each sequence.
    void iteration(std::function<std::vector<int>(const std::vector<std::vector<int>> &)> forward_batch);

    size_t active_slots() const;
    size_t pending() const;
    size_t max_batch_size() const { return max_batch_size_; }

   private:
    size_t max_batch_size_;
    std::vector<Slot> slots_;
    std::queue<Request> waiting_;
    mutable std::mutex mutex_;
};

}  // namespace server
}  // namespace serius
