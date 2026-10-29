#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "auto_tune.h"

namespace serius {
namespace automl {

/// Key type for caching tuning results: (M, N, K) dimensions.
using TuneCacheKey = std::tuple<int, int, int>;

/// Persistent cache for auto-tuning results.
/// Saves optimal configurations to disk and loads them on subsequent runs.
class TuneCache
{
   public:
    TuneCache();
    explicit TuneCache(const std::string &cache_file);

    /// Look up a cached configuration. Returns false if not found.
    bool get(const TuneCacheKey &key, KernelConfig &config, float &time_ms) const;

    /// Store a configuration in the cache.
    void put(const TuneCacheKey &key, const KernelConfig &config, float time_ms);

    /// Save the cache to disk (JSON format).
    bool save() const;

    /// Load the cache from disk.
    bool load();

    /// Clear all cached entries.
    void clear();

    /// Number of cached entries.
    size_t size() const { return cache_.size(); }

    const std::string &cache_file() const { return cache_file_; }

   private:
    std::string cache_file_;
    std::map<TuneCacheKey, std::pair<KernelConfig, float>> cache_;
};

}  // namespace automl
}  // namespace serius