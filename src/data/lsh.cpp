#include "lsh.h"

#include <cstdint>
#include <functional>

namespace serius {
namespace data {

LSHIndex::LSHIndex(int bands, int rows_per_band)
    : bands_(bands), rows_per_band_(rows_per_band), buckets_(static_cast<size_t>(bands))
{
}

size_t LSHIndex::band_hash(const std::vector<uint32_t> &signature, int band) const
{
    size_t start = static_cast<size_t>(band) * static_cast<size_t>(rows_per_band_);
    size_t end = std::min(start + static_cast<size_t>(rows_per_band_), signature.size());
    size_t h = 14695981039346656037ULL;
    for (size_t i = start; i < end; ++i) {
        h ^= signature[i];
        h *= 1099511628211ULL;
    }
    return h;
}

void LSHIndex::insert(int doc_id, const std::vector<uint32_t> &signature)
{
    for (int b = 0; b < bands_; ++b) {
        size_t bh = band_hash(signature, b);
        buckets_[static_cast<size_t>(b)][bh].push_back(doc_id);
    }
}

std::set<std::pair<int, int>> LSHIndex::find_candidates() const
{
    std::set<std::pair<int, int>> pairs;
    for (const auto &band : buckets_) {
        for (const auto &kv : band) {
            const auto &docs = kv.second;
            if (docs.size() > 1) {
                for (size_t i = 0; i < docs.size(); ++i) {
                    for (size_t j = i + 1; j < docs.size(); ++j) {
                        int a = docs[i];
                        int b = docs[j];
                        if (a > b) {
                            std::swap(a, b);
                        }
                        pairs.insert({a, b});
                    }
                }
            }
        }
    }
    return pairs;
}

}  // namespace data
}  // namespace serius
