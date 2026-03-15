#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tensor/tensor.h"
#include "transformer/paged_kv_cache.h"

namespace serius {
namespace test {

using core::Tensor;
using model::BlockManager;
using model::PagedKVCache;

TEST(PagedKVCacheTest, BlockAllocation)
{
    size_t num_blocks = 4;
    size_t block_size = 16;
    size_t num_heads = 2;
    size_t head_dim = 32;

    auto manager = std::make_shared<BlockManager>(num_blocks, block_size, num_heads, head_dim);

    EXPECT_EQ(manager->num_free_blocks(), 4);

    int b1 = manager->allocate();
    EXPECT_EQ(manager->num_free_blocks(), 3);
    EXPECT_EQ(b1, 0);

    int b2 = manager->allocate();
    EXPECT_EQ(manager->num_free_blocks(), 2);
    EXPECT_EQ(b2, 1);

    manager->free(b1);
    EXPECT_EQ(manager->num_free_blocks(), 3);

    int b3 = manager->allocate();
    EXPECT_EQ(b3, 0);  // Should reuse b1 (stack-based)
}

TEST(PagedKVCacheTest, PagingLogic)
{
    size_t num_blocks = 2;
    size_t block_size = 4;  // small block size for testing
    size_t num_heads = 1;
    size_t head_dim = 8;

    auto manager = std::make_shared<BlockManager>(num_blocks, block_size, num_heads, head_dim);
    PagedKVCache cache(manager);

    // Initial update: 6 tokens (should take 2 blocks)
    Tensor k({1, 6, 8});
    Tensor v({1, 6, 8});
    k.fill(1.0f);
    v.fill(2.0f);

    cache.update(k, v);

    EXPECT_EQ(cache.current_len(), 6);
    EXPECT_EQ(cache.num_blocks(), 2);
    EXPECT_EQ(manager->num_free_blocks(), 0);

    // Check data in first block (0)
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(manager->get_k_pool().item({0, 0, i, 0}), 1.0f);
        EXPECT_EQ(manager->get_v_pool().item({0, 0, i, 0}), 2.0f);
    }

    // Check data in second block (1)
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(manager->get_k_pool().item({1, 0, i, 0}), 1.0f);
        EXPECT_EQ(manager->get_v_pool().item({1, 0, i, 0}), 2.0f);
    }
}

TEST(PagedKVCacheTest, OverflowHandling)
{
    size_t num_blocks = 1;
    size_t block_size = 4;
    size_t num_heads = 1;
    size_t head_dim = 8;

    auto manager = std::make_shared<BlockManager>(num_blocks, block_size, num_heads, head_dim);
    PagedKVCache cache(manager);

    Tensor k({1, 5, 8});
    Tensor v({1, 5, 8});

    // Should throw because it needs 2 blocks but pool only has 1
    EXPECT_THROW(cache.update(k, v), std::runtime_error);
}

TEST(PagedKVCacheTest, GatherLogic)
{
    size_t num_blocks = 2;
    size_t block_size = 4;
    size_t num_heads = 1;
    size_t head_dim = 8;

    auto manager = std::make_shared<BlockManager>(num_blocks, block_size, num_heads, head_dim);
    PagedKVCache cache(manager);

    Tensor k_in({1, 6, 8});
    Tensor v_in({1, 6, 8});
    for (size_t i = 0; i < 6; ++i) {
        k_in.at(0, i, 0) = static_cast<float>(i);
        v_in.at(0, i, 0) = static_cast<float>(10 + i);
    }

    cache.update(k_in, v_in);

    Tensor gathered_k = cache.gather_keys();
    Tensor gathered_v = cache.gather_values();

    EXPECT_EQ(gathered_k.shape()[2], 6);
    EXPECT_EQ(gathered_v.shape()[2], 6);

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(gathered_k.at(0, 0, i, 0), static_cast<float>(i), 1e-5f);
        EXPECT_NEAR(gathered_v.at(0, 0, i, 0), static_cast<float>(10 + i), 1e-5f);
    }
}

}  // namespace test
}  // namespace serius
