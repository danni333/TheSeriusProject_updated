#include "prefix_cache.h"

#include <functional>

namespace serius {
namespace cache {

PrefixCache::PrefixCache(const Config& config)
    : config_(config)
{
}

std::pair<size_t, KVCacheSnapshot*> PrefixCache::find_longest_prefix(
    const std::vector<int>& tokens)
{
    // Try from longest to shortest prefix
    for (size_t len = tokens.size(); len > 0; --len) {
        uint64_t hash = hash_tokens(tokens.data(), len);
        auto it = cache_.find(hash);
        if (it != cache_.end()) {
            // Move to front (MRU)
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_it);
            return {len, &it->second.snapshot};
        }
    }
    return {0, nullptr};
}

void PrefixCache::store(const std::vector<int>& tokens, size_t prefix_len,
                       const KVCacheSnapshot& kv)
{
    if (prefix_len == 0 || prefix_len > tokens.size()) {
        return;
    }

    uint64_t hash = hash_tokens(tokens.data(), prefix_len);

    // Check if already exists
    auto it = cache_.find(hash);
    if (it != cache_.end()) {
        // Update existing entry and move to MRU
        memory_usage_ -= (it->second.snapshot.keys.size() * it->second.snapshot.keys.element_size() +
                          it->second.snapshot.values.size() * it->second.snapshot.values.element_size());
        it->second.snapshot = kv;
        memory_usage_ += (kv.keys.size() * kv.keys.element_size() + kv.values.size() * kv.values.element_size());
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_it);
        return;
    }

    // Evict if necessary
    while (cache_.size() >= config_.max_entries ||
           memory_usage_ + kv.keys.size() * kv.keys.element_size() + kv.values.size() * kv.values.element_size() > config_.max_memory_bytes) {
        if (cache_.empty()) {
            break;
        }
        evict_lru();
    }

    // Add new entry
    CacheEntry entry;
    entry.snapshot = kv;
    lru_list_.push_front(hash);
    entry.lru_it = lru_list_.begin();
    cache_[hash] = std::move(entry);
    memory_usage_ += (kv.keys.size() * kv.keys.element_size() + kv.values.size() * kv.values.element_size());
}

void PrefixCache::clear()
{
    cache_.clear();
    lru_list_.clear();
    memory_usage_ = 0;
}

uint64_t PrefixCache::hash_tokens(const int* tokens, size_t len) const
{
    // Simple FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(tokens[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void PrefixCache::evict_lru()
{
    if (lru_list_.empty()) {
        return;
    }

    uint64_t hash = lru_list_.back();
    auto it = cache_.find(hash);
    if (it != cache_.end()) {
        memory_usage_ -= (it->second.snapshot.keys.size() * it->second.snapshot.keys.element_size() +
                          it->second.snapshot.values.size() * it->second.snapshot.values.element_size());
        cache_.erase(it);
    }
    lru_list_.pop_back();
}

}  // namespace cache
}  // namespace serius
