#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "minhash.h"
#include "lsh.h"

namespace serius {
namespace data {

struct Document
{
    int id;
    std::string text;
};

/// Result of running the deduplication pipeline.
struct DedupResult
{
    std::unordered_set<int> kept_ids;       // document ids to keep
    std::unordered_set<int> removed_ids;    // document ids removed as duplicates
    int duplicate_pairs = 0;                // number of candidate pairs found
    int removed_count = 0;                  // number of documents removed
};

/// End-to-end deduplication: MinHash + LSH + Jaccard verification.
class DedupPipeline
{
   public:
    DedupPipeline(int num_hashes = 128, int bands = 16, int rows_per_band = 8,
                  float jaccard_threshold = 0.8f, int ngram = 5);

    /// Run deduplication over a collection of documents.
    DedupResult run(const std::vector<Document> &docs);

   private:
    int num_hashes_;
    int bands_;
    int rows_per_band_;
    float jaccard_threshold_;
    int ngram_;
};

}  // namespace data
}  // namespace serius
