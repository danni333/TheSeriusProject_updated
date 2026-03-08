#pragma once

#include <memory>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace transformer {

class KVCache
{
   public:
    KVCache(size_t max_seq_len, size_t num_heads, size_t head_dim, bool rolling = false);
    ~KVCache() = default;

    /// Update the cache with new keys and values for the current step
    /// @param k: [batch_size, num_heads, new_seq_len, head_dim]
    /// @param v: [batch_size, num_heads, new_seq_len, head_dim]
    void update(const Tensor &k, const Tensor &v);

    /// Get the current cached keys
    /// @return [batch_size, num_heads, total_seq_len, head_dim]
    Tensor get_keys() const;

    /// Get the current cached values
    /// @return [batch_size, num_heads, total_seq_len, head_dim]
    Tensor get_values() const;

    void reset();

    /// Reorder cache entries for beam search
    /// @param beam_indices Maps new beam index → old beam index
    void reorder(const std::vector<int> &beam_indices);

    size_t current_len() const { return current_len_; }
    size_t max_len() const { return max_len_; }
    bool is_rolling() const { return rolling_; }

   private:
    std::unique_ptr<Tensor> keys_;
    std::unique_ptr<Tensor> values_;
    size_t max_len_;
    size_t current_len_;
    size_t num_heads_;
    size_t head_dim_;
    bool rolling_;
};

}  // namespace transformer
}  // namespace serius
