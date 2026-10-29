#include "minhash.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace serius {
namespace data {

namespace {

constexpr uint64_t kFNVOffset = 14695981039346656037ULL;
constexpr uint64_t kFNVPrime = 1099511628211ULL;

std::vector<std::string> extract_ngrams(const std::string &text, int ngram)
{
    std::vector<std::string> shingles;
    if (ngram <= 0 || static_cast<size_t>(ngram) > text.size()) {
        return shingles;
    }
    for (size_t i = 0; i + static_cast<size_t>(ngram) <= text.size(); ++i) {
        shingles.push_back(text.substr(i, static_cast<size_t>(ngram)));
    }
    return shingles;
}

}  // namespace

uint64_t hash64(const std::string &s)
{
    uint64_t h = kFNVOffset;
    for (unsigned char c : s) {
        h ^= c;
        h *= kFNVPrime;
    }
    return h;
}

std::vector<uint32_t> minhash_signature(const std::string &text,
                                        int num_hashes,
                                        int ngram)
{
    std::vector<uint32_t> signature(static_cast<size_t>(num_hashes), UINT32_MAX);
    if (num_hashes <= 0) {
        return signature;
    }

    // Deterministic hash parameters (a, b) per hash function.
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint64_t> dist_a(1, UINT64_MAX);
    std::uniform_int_distribution<uint64_t> dist_b(0, UINT64_MAX);
    std::vector<uint64_t> a(num_hashes), b(num_hashes);
    for (int i = 0; i < num_hashes; ++i) {
        a[i] = dist_a(gen);
        b[i] = dist_b(gen);
    }

    auto shingles = extract_ngrams(text, ngram);
    for (const auto &shingle : shingles) {
        uint64_t h = hash64(shingle);
        for (int i = 0; i < num_hashes; ++i) {
            uint32_t val = static_cast<uint32_t>((a[i] * h + b[i]) >> 32);
            if (val < signature[static_cast<size_t>(i)]) {
                signature[static_cast<size_t>(i)] = val;
            }
        }
    }
    return signature;
}

float minhash_jaccard(const std::vector<uint32_t> &a,
                      const std::vector<uint32_t> &b)
{
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0f;
    }
    size_t matches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) {
            ++matches;
        }
    }
    return static_cast<float>(matches) / static_cast<float>(a.size());
}

}  // namespace data
}  // namespace serius
