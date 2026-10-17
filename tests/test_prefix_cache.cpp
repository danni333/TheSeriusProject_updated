#include <gtest/gtest.h>

#include "cache/prefix_cache.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::cache;
using namespace serius::core;

TEST(PrefixCache, EmptyCache)
{
    PrefixCache cache;
    std::vector<int> tokens = {1, 2, 3, 4};

    auto [len, snapshot] = cache.find_longest_prefix(tokens);
    EXPECT_EQ(len, 0);
    EXPECT_EQ(snapshot, nullptr);
    EXPECT_EQ(cache.size(), 0);
}

TEST(PrefixCache, StoreAndRetrieve)
{
    PrefixCache cache;
    std::vector<int> tokens = {1, 2, 3, 4, 5};

    // Create a snapshot
    Tensor keys({2, 4, 8});
    Tensor values({2, 4, 8});
    keys.fill(1.0f);
    values.fill(2.0f);
    KVCacheSnapshot snapshot(std::move(keys), std::move(values), 3);

    // Store prefix of length 3
    cache.store(tokens, 3, snapshot);

    EXPECT_EQ(cache.size(), 1);

    // Retrieve
    auto [len, retrieved] = cache.find_longest_prefix(tokens);
    EXPECT_EQ(len, 3);
    EXPECT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->seq_len, 3);
}

TEST(PrefixCache, LongestPrefixMatch)
{
    PrefixCache cache;

    // Store prefix of length 3
    std::vector<int> tokens1 = {1, 2, 3, 4, 5};
    Tensor k1({1, 1, 1});
    Tensor v1({1, 1, 1});
    cache.store(tokens1, 3, KVCacheSnapshot(std::move(k1), std::move(v1), 3));

    // Store prefix of length 5 (different tokens after position 3)
    std::vector<int> tokens2 = {1, 2, 3, 6, 7};
    Tensor k2({1, 1, 1});
    Tensor v2({1, 1, 1});
    cache.store(tokens2, 5, KVCacheSnapshot(std::move(k2), std::move(v2), 5));

    // Query with tokens matching prefix of length 3
    std::vector<int> query = {1, 2, 3, 8, 9};
    auto [len, snapshot] = cache.find_longest_prefix(query);
    EXPECT_EQ(len, 3);
}

TEST(PrefixCache, Miss)
{
    PrefixCache cache;

    std::vector<int> tokens1 = {1, 2, 3};
    Tensor k1({1, 1, 1});
    Tensor v1({1, 1, 1});
    cache.store(tokens1, 3, KVCacheSnapshot(std::move(k1), std::move(v1), 3));

    // Query with completely different tokens
    std::vector<int> query = {4, 5, 6};
    auto [len, snapshot] = cache.find_longest_prefix(query);
    EXPECT_EQ(len, 0);
    EXPECT_EQ(snapshot, nullptr);
}

TEST(PrefixCache, UpdateExisting)
{
    PrefixCache cache;

    std::vector<int> tokens = {1, 2, 3};
    Tensor k1({1, 1, 1});
    Tensor v1({1, 1, 1});
    k1.fill(1.0f);
    v1.fill(1.0f);
    cache.store(tokens, 3, KVCacheSnapshot(std::move(k1), std::move(v1), 3));

    EXPECT_EQ(cache.size(), 1);

    // Update with new snapshot
    Tensor k2({1, 1, 1});
    Tensor v2({1, 1, 1});
    k2.fill(2.0f);
    v2.fill(2.0f);
    cache.store(tokens, 3, KVCacheSnapshot(std::move(k2), std::move(v2), 3));

    // Should still be 1 entry
    EXPECT_EQ(cache.size(), 1);

    // Retrieved snapshot should have new values
    auto [len, snapshot] = cache.find_longest_prefix(tokens);
    EXPECT_NE(snapshot, nullptr);
    EXPECT_FLOAT_EQ(snapshot->keys.at(0, 0, 0), 2.0f);
}

TEST(PrefixCache, EvictionLRU)
{
    PrefixCache::Config cfg;
    cfg.max_entries = 2;
    PrefixCache cache(cfg);

    std::vector<int> tokens1 = {1, 2, 3};
    Tensor k1({1, 1, 1});
    Tensor v1({1, 1, 1});
    cache.store(tokens1, 3, KVCacheSnapshot(std::move(k1), std::move(v1), 3));

    std::vector<int> tokens2 = {4, 5, 6};
    Tensor k2({1, 1, 1});
    Tensor v2({1, 1, 1});
    cache.store(tokens2, 3, KVCacheSnapshot(std::move(k2), std::move(v2), 3));

    EXPECT_EQ(cache.size(), 2);

    // Access tokens1 to make it MRU
    cache.find_longest_prefix(tokens1);

    // Add third entry - should evict tokens2 (LRU)
    std::vector<int> tokens3 = {7, 8, 9};
    Tensor k3({1, 1, 1});
    Tensor v3({1, 1, 1});
    cache.store(tokens3, 3, KVCacheSnapshot(std::move(k3), std::move(v3), 3));

    EXPECT_EQ(cache.size(), 2);

    // tokens1 should still be present
    auto [len1, snap1] = cache.find_longest_prefix(tokens1);
    EXPECT_EQ(len1, 3);

    // tokens2 should be evicted
    auto [len2, snap2] = cache.find_longest_prefix(tokens2);
    EXPECT_EQ(len2, 0);
}

TEST(PrefixCache, MemoryTracking)
{
    PrefixCache::Config cfg;
    cfg.max_memory_bytes = 1000;
    PrefixCache cache(cfg);

    std::vector<int> tokens = {1, 2, 3};

    // Create snapshot with known size
    Tensor keys({2, 4, 8});  // 2*4*8*4 = 256 bytes
    Tensor values({2, 4, 8});  // 256 bytes
    keys.fill(1.0f);
    values.fill(1.0f);

    cache.store(tokens, 3, KVCacheSnapshot(std::move(keys), std::move(values), 3));

    size_t expected_memory = (2 * 4 * 8 * keys.element_size()) + (2 * 4 * 8 * values.element_size());
    EXPECT_EQ(cache.memory_usage(), expected_memory);
}

TEST(PrefixCache, Clear)
{
    PrefixCache cache;

    std::vector<int> tokens = {1, 2, 3};
    Tensor k({1, 1, 1});
    Tensor v({1, 1, 1});
    cache.store(tokens, 3, KVCacheSnapshot(std::move(k), std::move(v), 3));

    EXPECT_EQ(cache.size(), 1);
    EXPECT_GT(cache.memory_usage(), 0);

    cache.clear();

    EXPECT_EQ(cache.size(), 0);
    EXPECT_EQ(cache.memory_usage(), 0);
}

TEST(PrefixCache, ZeroPrefixLength)
{
    PrefixCache cache;

    std::vector<int> tokens = {1, 2, 3};
    Tensor k({1, 1, 1});
    Tensor v({1, 1, 1});

    // Storing with prefix_len = 0 should do nothing
    cache.store(tokens, 0, KVCacheSnapshot(std::move(k), std::move(v), 3));

    EXPECT_EQ(cache.size(), 0);
}

TEST(PrefixCache, PrefixLongerThanTokens)
{
    PrefixCache cache;

    std::vector<int> tokens = {1, 2, 3};
    Tensor k({1, 1, 1});
    Tensor v({1, 1, 1});

    // Storing with prefix_len > tokens.size() should do nothing
    cache.store(tokens, 5, KVCacheSnapshot(std::move(k), std::move(v), 3));

    EXPECT_EQ(cache.size(), 0);
}
