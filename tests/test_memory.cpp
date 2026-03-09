#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "memory/memory_pool.h"
#include "memory/pool_allocator.h"
#include "tensor/tensor.h"

namespace serius {
namespace test {

namespace core = serius::core;
using core::Tensor;
using core::memory::MemoryPool;
using core::memory::PoolAllocator;
using core::memory::ScopedArena;

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

TEST(MemoryPoolTest, StatsTracking)
{
    MemoryPool pool(1024);

    EXPECT_EQ(pool.peak_used(), 0);
    EXPECT_EQ(pool.allocation_count(), 0);

    pool.allocate(128);
    EXPECT_EQ(pool.peak_used(), 128);
    EXPECT_EQ(pool.allocation_count(), 1);

    pool.allocate(256);
    EXPECT_EQ(pool.peak_used(), 384);
    EXPECT_EQ(pool.allocation_count(), 2);

    pool.reset();
    EXPECT_EQ(pool.peak_used(), 384);  // Peak should not shrink on reset

    pool.allocate(64);
    EXPECT_EQ(pool.peak_used(), 384);  // Current used is 64, peak is still 384
    EXPECT_EQ(pool.allocation_count(), 3);
}

TEST(MemoryPoolTest, PrintStats)
{
    MemoryPool pool(1024);
    pool.allocate(128);
    // Just verifying it doesn't crash or throw
    EXPECT_NO_THROW(pool.print_stats());
}

TEST(MemoryPoolTest, ThreadLocalPool)
{
    // Ensure the main thread has access to a working pool
    MemoryPool &main_pool = MemoryPool::get_thread_local();
    void *ptr1 = main_pool.allocate(128);
    EXPECT_NE(ptr1, nullptr);
    size_t main_used = main_pool.used();
    EXPECT_EQ(main_used, 128);

    // Let's test thread independence if OpenMP is enabled,
    // otherwise we just tested that get_thread_local() compiles and works.
#ifdef _OPENMP
    bool thread_independent = true;
#pragma omp parallel num_threads(2)
    {
        MemoryPool &thread_pool = MemoryPool::get_thread_local();
        void *t_ptr = thread_pool.allocate(256);
        EXPECT_NE(t_ptr, nullptr);

        // At this point each thread should have used 256 bytes in its own local pool
        // The main thread's local pool might be used since it can participate in the thread team
        if (omp_get_thread_num() != 0) {
            if (thread_pool.used() != 256) {
                thread_independent = false;
            }
        }
    }
    EXPECT_TRUE(thread_independent);
#endif
}

}  // namespace test
}  // namespace serius
