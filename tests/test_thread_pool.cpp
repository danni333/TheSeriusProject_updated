#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <vector>

#include "../src/common/thread_pool.h"

// ----- ThreadPool basic tests -----

TEST(ThreadPoolTest, InstanceReturnsPool)
{
    auto &pool = serius::ThreadPool::instance();
    EXPECT_GT(pool.num_threads(), 0u);
}

TEST(ThreadPoolTest, SubmitAndWait)
{
    auto &pool = serius::ThreadPool::instance();
    std::atomic<int> counter{0};

    constexpr int NUM_TASKS = 100;
    std::vector<std::future<void>> futures;
    futures.reserve(NUM_TASKS);

    for (int i = 0; i < NUM_TASKS; ++i) {
        futures.push_back(pool.submit([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); }));
    }

    for (auto &fut : futures) {
        fut.get();
    }

    EXPECT_EQ(counter.load(), NUM_TASKS);
}

// ----- parallel_for tests -----

TEST(ParallelForTest, EmptyRange)
{
    // Should not crash
    serius::parallel_for(size_t(5), size_t(5), [](size_t) { FAIL() << "Should never be called"; });
    serius::parallel_for(size_t(10), size_t(5), [](size_t) { FAIL() << "Should never be called"; });
}

TEST(ParallelForTest, SingleElement)
{
    std::atomic<int> count{0};
    serius::parallel_for(size_t(0), size_t(1), [&count](size_t) { count.fetch_add(1); });
    EXPECT_EQ(count.load(), 1);
}

TEST(ParallelForTest, AllIndicesVisited)
{
    constexpr size_t NUM = 1000;
    std::vector<std::atomic<int>> visited(NUM);
    for (auto &val : visited) {
        val.store(0);
    }

    serius::parallel_for(size_t(0), NUM, [&visited](size_t idx) { visited[idx].fetch_add(1); });

    for (size_t idx = 0; idx < NUM; ++idx) {
        EXPECT_EQ(visited[idx].load(), 1) << "Index " << idx << " visited wrong number of times";
    }
}

TEST(ParallelForTest, CorrectResults)
{
    // Compute sum of squares in parallel
    constexpr size_t NUM = 10000;
    std::vector<double> results(NUM, 0.0);

    serius::parallel_for(size_t(0), NUM, [&results](size_t idx) {
        results[idx] = static_cast<double>(idx) * static_cast<double>(idx);
    });

    // Verify
    for (size_t idx = 0; idx < NUM; ++idx) {
        EXPECT_DOUBLE_EQ(results[idx], static_cast<double>(idx) * static_cast<double>(idx));
    }
}

// ----- parallel_for_chunked tests -----

TEST(ParallelForChunkedTest, AllIndicesCovered)
{
    constexpr size_t NUM = 500;
    std::vector<std::atomic<int>> visited(NUM);
    for (auto &val : visited) {
        val.store(0);
    }

    serius::parallel_for_chunked(size_t(0), NUM, [&visited](size_t lo, size_t hi) {
        for (size_t idx = lo; idx < hi; ++idx) {
            visited[idx].fetch_add(1);
        }
    });

    for (size_t idx = 0; idx < NUM; ++idx) {
        EXPECT_EQ(visited[idx].load(), 1) << "Index " << idx << " visited wrong number of times";
    }
}

// ----- Performance sanity check -----

TEST(ParallelForTest, FasterThanSequentialForLargeWork)
{
    // This is a sanity check, not a strict benchmark.
    // We do enough work per iteration to make threading worthwhile.
    constexpr size_t NUM = 256;
    std::vector<double> data(NUM * NUM, 1.0);

    // Sequential timing
    auto start_seq = std::chrono::high_resolution_clock::now();
    for (size_t idx = 0; idx < NUM; ++idx) {
        double sum = 0.0;
        for (size_t jdx = 0; jdx < NUM; ++jdx) {
            for (size_t kdx = 0; kdx < NUM; ++kdx) {
                sum += data[jdx * NUM + kdx] * 0.001;
            }
        }
        data[idx] = sum;
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    auto seq_us = std::chrono::duration_cast<std::chrono::microseconds>(end_seq - start_seq).count();

    // Reset
    std::fill(data.begin(), data.end(), 1.0);

    // Parallel timing
    auto start_par = std::chrono::high_resolution_clock::now();
    serius::parallel_for(size_t(0), NUM, [&data, NUM](size_t idx) {
        double sum = 0.0;
        for (size_t jdx = 0; jdx < NUM; ++jdx) {
            for (size_t kdx = 0; kdx < NUM; ++kdx) {
                sum += data[jdx * NUM + kdx] * 0.001;
            }
        }
        data[idx] = sum;
    });
    auto end_par = std::chrono::high_resolution_clock::now();
    auto par_us = std::chrono::duration_cast<std::chrono::microseconds>(end_par - start_par).count();

    // Parallel should be at least somewhat faster on multi-core machines
    // (relaxed check: parallel <= 90% of sequential, or sequential was very fast)
    if (seq_us > 1000) {
        // Only check speedup if sequential took a meaningful amount of time
        EXPECT_LT(par_us, seq_us) << "Parallel was not faster than sequential. "
                                   << "Sequential: " << seq_us << "us, Parallel: " << par_us << "us";
    }
}
