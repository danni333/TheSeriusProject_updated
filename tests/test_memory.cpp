#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "../src/memory/memory_pool.h"
#include "../src/memory/pool_allocator.h"


namespace serius {
namespace memory {
namespace test {

TEST(MemoryPoolTest, BasicAllocation)
{
    MemoryPool pool(1024);  // 1KB initial size

    void *ptr1 = pool.allocate(128);
    EXPECT_NE(ptr1, nullptr);

    void *ptr2 = pool.allocate(256);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);

    EXPECT_GT(pool.used(), 0);
    EXPECT_GE(pool.capacity(), 1024);
}

TEST(MemoryPoolTest, Alignment)
{
    MemoryPool pool;

    void *ptr1 = pool.allocate(13);
    void *ptr2 = pool.allocate(42);

    // Check if pointers are aligned to MemoryPool::ALIGNMENT (64)
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) % MemoryPool::ALIGNMENT, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) % MemoryPool::ALIGNMENT, 0);
}

TEST(MemoryPoolTest, Growth)
{
    MemoryPool pool(64);  // Small initial size

    void *ptr1 = pool.allocate(64);
    EXPECT_NE(ptr1, nullptr);

    // This should trigger a growth
    void *ptr2 = pool.allocate(128);
    EXPECT_NE(ptr2, nullptr);

    EXPECT_GT(pool.capacity(), 64);
}

TEST(MemoryPoolTest, Reset)
{
    MemoryPool pool(1024);

    pool.allocate(512);
    size_t used_before = pool.used();

    pool.reset();
    size_t used_after = pool.used();

    EXPECT_GT(used_before, 0);
    EXPECT_EQ(used_after, 0);

    // After reset, we should be able to allocate from the beginning
    void *ptr = pool.allocate(512);
    EXPECT_NE(ptr, nullptr);
}

TEST(MemoryPoolTest, ScopedArena)
{
    MemoryPool pool(1024);

    pool.allocate(128);
    size_t used_initial = pool.used();

    {
        ScopedArena arena(pool);
        pool.allocate(256);
        EXPECT_GT(pool.used(), used_initial);
    }

    // ScopedArena destructor should have reset the pool to the initial state
    EXPECT_EQ(pool.used(), used_initial);
}

TEST(MemoryPoolTest, PoolAllocatorBasic)
{
    MemoryPool pool(4096);
    PoolAllocator<int> allocator(pool);

    std::vector<int, PoolAllocator<int>> vec(allocator);

    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }

    EXPECT_EQ(vec.size(), 100);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(vec[i], i);
    }

    EXPECT_GT(pool.used(), 0);
}

}  // namespace test
}  // namespace memory
}  // namespace serius
