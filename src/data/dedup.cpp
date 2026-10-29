#include "dedup.h"

#include <algorithm>

namespace serius {
namespace data {

DedupPipeline::DedupPipeline(int num_hashes, int bands, int rows_per_band,
                             float jaccard_threshold, int ngram)
    : num_hashes_(num_hashes),
      bands_(bands),
      rows_per_band_(rows_per_band),
      jaccard_threshold_(jaccard_threshold),
      ngram_(ngram)
{
}

DedupResult DedupPipeline::run(const std::vector<Document> &docs)
{
    DedupResult result;

    // Compute MinHash signatures for all documents.
    std::unordered_map<int, std::vector<uint32_t>> signatures;
    signatures.reserve(docs.size());
    for (const auto &doc : docs) {
        signatures[doc.id] = minhash_signature(doc.text, num_hashes_, ngram_);
    }

    // Build LSH index and find candidate duplicate pairs.
    LSHIndex lsh(bands_, rows_per_band_);
    for (const auto &kv : signatures) {
        lsh.insert(kv.first, kv.second);
    }
    auto candidates = lsh.find_candidates();
    result.duplicate_pairs = static_cast<int>(candidates.size());

    // Verify candidates with exact Jaccard and mark duplicates for removal.
    // Keep the first occurrence of each duplicate group.
    std::unordered_set<int> removed;
    for (const auto &pair : candidates) {
        int a = pair.first;
        int b = pair.second;
        if (removed.count(a) || removed.count(b)) {
            continue;
        }
        float j = minhash_jaccard(signatures.at(a), signatures.at(b));
        if (j >= jaccard_threshold_) {
            // Remove b (keep the earlier document a).
            removed.insert(b);
            ++result.removed_count;
        }
    }

    for (const auto &doc : docs) {
        if (removed.count(doc.id)) {
            result.removed_ids.insert(doc.id);
        } else {
            result.kept_ids.insert(doc.id);
        }
    }
    return result;
}

}  // namespace data
}  // namespace serius
