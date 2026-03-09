#include "kv_cache.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace serius {
namespace model {

using core::Tensor;

KVCache::KVCache(size_t max_seq_len, size_t num_heads, size_t head_dim, bool rolling)
    : max_len_(max_seq_len), current_len_(0), num_heads_(num_heads), head_dim_(head_dim), rolling_(rolling)
{
    // Pre-allocate buffers for keys and values
    // Assuming a default batch size of 1 for now, or we'll reshape on first update
}

void KVCache::update(const Tensor &key, const Tensor &value)
{
    size_t batch_size = key.shape()[0];
    size_t num_heads = key.shape()[1];
    size_t new_len = key.shape()[2];
    size_t head_dim = key.shape()[3];

    if (keys_ == nullptr) {
        // Initialize buffers with correct batch size
        keys_ = std::make_unique<Tensor>(std::vector<size_t>{batch_size, num_heads, max_len_, head_dim});
        values_ = std::make_unique<Tensor>(std::vector<size_t>{batch_size, num_heads, max_len_, head_dim});
    }

    if (rolling_) {
        // Rolling buffer: overwrite circularly when full
        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                for (size_t seq_idx = 0; seq_idx < new_len; ++seq_idx) {
                    size_t pos_idx = (current_len_ + seq_idx) % max_len_;
                    for (size_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                        keys_->at({batch_idx, head_idx, pos_idx, d_idx}) =
                            key.at({batch_idx, head_idx, seq_idx, d_idx});
                        values_->at({batch_idx, head_idx, pos_idx, d_idx}) =
                            value.at({batch_idx, head_idx, seq_idx, d_idx});
                    }
                }
            }
        }
        current_len_ += new_len;
    } else {
        if (current_len_ + new_len > max_len_) {
            throw std::runtime_error("KV Cache overflow: maximum sequence length reached");
        }

        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                for (size_t seq_idx = 0; seq_idx < new_len; ++seq_idx) {
                    for (size_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                        keys_->at({batch_idx, head_idx, current_len_ + seq_idx, d_idx}) =
                            key.at({batch_idx, head_idx, seq_idx, d_idx});
                        values_->at({batch_idx, head_idx, current_len_ + seq_idx, d_idx}) =
                            value.at({batch_idx, head_idx, seq_idx, d_idx});
                    }
                }
            }
        }
        current_len_ += new_len;
    }
}

Tensor KVCache::get_keys() const
{
    if (keys_ == nullptr) return Tensor({0});
    size_t effective_len = rolling_ ? std::min(current_len_, max_len_) : current_len_;
    // Return a slice [batch, heads, 0:effective_len, head_dim]
    return keys_->slice(2, 0, effective_len);
}

Tensor KVCache::get_values() const
{
    if (values_ == nullptr) return Tensor({0});
    size_t effective_len = rolling_ ? std::min(current_len_, max_len_) : current_len_;
    // Return a slice [batch, heads, 0:effective_len, head_dim]
    return values_->slice(2, 0, effective_len);
}

void KVCache::reset()
{
    current_len_ = 0;
}

void KVCache::reorder(const std::vector<int> &beam_indices)
{
    if (keys_ == nullptr) return;

    size_t batch_size = beam_indices.size();
    size_t num_heads = keys_->shape()[1];
    size_t seq_len = rolling_ ? std::min(current_len_, max_len_) : current_len_;
    size_t head_dim = keys_->shape()[3];

    // Create temporary copies to reorder from
    Tensor keys_copy = keys_->copy();
    Tensor values_copy = values_->copy();

    for (size_t b = 0; b < batch_size; ++b) {
        int src = beam_indices[b];
        if (src < 0 || static_cast<size_t>(src) >= batch_size) {
            throw std::runtime_error("KVCache::reorder: invalid beam index");
        }
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t t = 0; t < seq_len; ++t) {
                for (size_t d = 0; d < head_dim; ++d) {
                    keys_->at({b, h, t, d}) = keys_copy.at({static_cast<size_t>(src), h, t, d});
                    values_->at({b, h, t, d}) = values_copy.at({static_cast<size_t>(src), h, t, d});
                }
            }
        }
    }
}

}  // namespace model
}  // namespace serius
