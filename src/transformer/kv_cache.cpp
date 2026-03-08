#include "kv_cache.h"

#include <cstring>
#include <stdexcept>


namespace serius {
namespace transformer {

KVCache::KVCache(size_t max_seq_len, size_t num_heads, size_t head_dim)
    : max_len_(max_seq_len), current_len_(0), num_heads_(num_heads), head_dim_(head_dim)
{
    // Pre-allocate buffers for keys and values
    // Assuming a default batch size of 1 for now, or we'll reshape on first update
}

void KVCache::update(const Tensor &k, const Tensor &v)
{
    size_t batch_size = k.shape()[0];
    size_t num_heads = k.shape()[1];
    size_t new_len = k.shape()[2];
    size_t head_dim = k.shape()[3];

    if (current_len_ + new_len > max_len_) {
        throw std::runtime_error("KV Cache overflow: maximum sequence length reached");
    }

    if (keys_ == nullptr) {
        // Initialize buffers with correct batch size
        keys_ = std::make_unique<Tensor>(std::vector<size_t>{batch_size, num_heads, max_len_, head_dim});
        values_ = std::make_unique<Tensor>(std::vector<size_t>{batch_size, num_heads, max_len_, head_dim});
    }

    // copy into pre-allocated slice
    // Tensor::slice returns a view, but we need to actually copy data into the pre-allocated buffer
    // For now, let's use element-wise copy since we don't have a specific slice-copy-into method

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t t = 0; t < new_len; ++t) {
                for (size_t d = 0; d < head_dim; ++d) {
                    keys_->at({b, h, current_len_ + t, d}) = k.at({b, h, t, d});
                    values_->at({b, h, current_len_ + t, d}) = v.at({b, h, t, d});
                }
            }
        }
    }

    current_len_ += new_len;
}

Tensor KVCache::get_keys() const
{
    if (keys_ == nullptr) return Tensor({0});
    // Return a slice [batch, heads, 0:current_len, head_dim]
    return keys_->slice(2, 0, current_len_);
}

Tensor KVCache::get_values() const
{
    if (values_ == nullptr) return Tensor({0});
    // Return a slice [batch, heads, 0:current_len, head_dim]
    return values_->slice(2, 0, current_len_);
}

void KVCache::reset()
{
    current_len_ = 0;
}

}  // namespace transformer
}  // namespace serius
