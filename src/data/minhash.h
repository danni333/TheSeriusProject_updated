#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace serius {
namespace data {

/// 64-bit hash per shingle (FNV-1a style).
uint64_t hash64(const std::string &s);

/// MinHash signature of a document.
/// Returns a vector of num_hashes minima.
std::vector<uint32_t> minhash_signature(const std::string &text,
                                        int num_hashes,
                                        int ngram = 5);

/// Jaccard similarity between two MinHash signatures.
float minhash_jaccard(const std::vector<uint32_t> &a,
                      const std::vector<uint32_t> &b);

}  // namespace data
}  // namespace serius
