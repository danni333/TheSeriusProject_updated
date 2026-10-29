#pragma once

#include <cstddef>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "minhash.h"

namespace serius {
namespace data {

/// Locality-Sensitive Hashing index over MinHash signatures.
/// Splits each signature into `bands` bands of `rows_per_band` rows.
/// Documents sharing a band hash become candidate duplicate pairs.
class LSHIndex
{
   public:
    LSHIndex(int bands, int rows_per_band);

    /// Insert a document signature.
    void insert(int doc_id, const std::vector<uint32_t> &signature);

    /// Find candidate duplicate pairs (doc_id, doc_id).
    std::set<std::pair<int, int>> find_candidates() const;

   private:
    int bands_;
    int rows_per_band_;

    // band_index -> band_hash -> set of doc_ids
    std::vector<std::unordered_map<size_t, std::vector<int>>> buckets_;

    size_t band_hash(const std::vector<uint32_t> &signature, int band) const;
};

}  // namespace data
}  // namespace serius
