#include "tune_cache.h"

#include <algorithm>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace serius {
namespace automl {

TuneCache::TuneCache() : cache_file_("tuning_cache.json")
{
}

TuneCache::TuneCache(const std::string &cache_file) : cache_file_(cache_file)
{
}

bool TuneCache::get(const TuneCacheKey &key, KernelConfig &config, float &time_ms) const
{
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return false;
    }
    config = it->second.first;
    time_ms = it->second.second;
    return true;
}

void TuneCache::put(const TuneCacheKey &key, const KernelConfig &config, float time_ms)
{
    cache_[key] = std::make_pair(config, time_ms);
}

bool TuneCache::save() const
{
    std::ofstream out(cache_file_);
    if (!out.is_open()) {
        return false;
    }

    nlohmann::json db = nlohmann::json::array();
    for (const auto &[key, value] : cache_) {
        nlohmann::json entry;
        entry["M"] = std::get<0>(key);
        entry["N"] = std::get<1>(key);
        entry["K"] = std::get<2>(key);
        entry["block_x"] = value.first.block_x;
        entry["block_y"] = value.first.block_y;
        entry["tile_size"] = value.first.tile_size;
        entry["use_shared_mem"] = value.first.use_shared_mem;
        entry["time_ms"] = value.second;
        db.push_back(entry);
    }

    out << db.dump(2);
    return true;
}

bool TuneCache::load()
{
    std::ifstream in(cache_file_);
    if (!in.is_open()) {
        return false;
    }

    try {
        nlohmann::json db;
        in >> db;

        if (!db.is_array()) {
            return false;
        }

        for (const auto &entry : db) {
            TuneCacheKey key = std::make_tuple(entry["M"], entry["N"], entry["K"]);
            KernelConfig cfg;
            cfg.block_x = entry["block_x"];
            cfg.block_y = entry["block_y"];
            cfg.tile_size = entry["tile_size"];
            cfg.use_shared_mem = entry["use_shared_mem"];
            float t = entry["time_ms"];
            cache_[key] = std::make_pair(cfg, t);
        }
        return true;
    } catch (...) {
        return false;
    }
}

void TuneCache::clear()
{
    cache_.clear();
}

}  // namespace automl
}  // namespace serius