#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace cache {

using core::Tensor;

/// Snapshot of KV cache for a specific prefix.
struct KVCacheSnapshot {
    Tensor keys;
    Tensor values;
    size_t seq_len;

    KVCacheSnapshot() : seq_len(0) {}
    KVCacheSnapshot(Tensor k, Tensor v, size_t len)
        : keys(std::move(k)), values(std::move(v)), seq_len(len) {}
};

/// Prefix cache for KV cache reuse.
/// Stores KV cache snapshots of common prefixes (system prompts, RAG context).
/// Uses LRU eviction to manage memory.
class PrefixCache
{
public:
    struct Config {
        size_t max_entries = 64;
        size_t max_memory_bytes = 1ULL << 30;  // 1GB default
    };

    explicit PrefixCache(const Config& config = Config{});

    /// Finds the longest matching prefix in the cache.
    /// @param tokens Token sequence to search for
    /// @return Pair of (prefix_length, snapshot_ptr). If no match, returns (0, nullptr).
    std::pair<size_t, KVCacheSnapshot*> find_longest_prefix(const std::vector<int>& tokens);

    /// Stores a KV cache snapshot for a prefix.
    /// @param tokens Token sequence
    /// @param prefix_len Length of the prefix to cache
    /// @param kv KV cache snapshot
    void store(const std::vector<int>& tokens, size_t prefix_len,
               const KVCacheSnapshot& kv);

    /// Clears all cached entries.
    void clear();

    /// Returns current number of cached entries.
    size_t size() const { return cache_.size(); }

    /// Returns current memory usage in bytes.
    size_t memory_usage() const { return memory_usage_; }

private:
    struct CacheEntry {
        KVCacheSnapshot snapshot;
        std::list<uint64_t>::iterator lru_it;
    };

    uint64_t hash_tokens(const int* tokens, size_t len) const;

    void evict_lru();

    Config config_;
    std::unordered_map<uint64_t, CacheEntry> cache_;
    std::list<uint64_t> lru_list_;  // MRU at front, LRU at back
    size_t memory_usage_ = 0;
};

}  // namespace cache
}  // namespace serius
