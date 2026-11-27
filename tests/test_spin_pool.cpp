// Tests for serius::SpinPool (Fase 73e, persistent ggml-style thread pool).

#include <gtest/gtest.h>

#include <atomic>
#include <numeric>
#include <vector>

#include "common/spin_pool.h"

using namespace serius;

TEST(SpinPool, SumMatchesSerial)
{
    auto &pool = SpinPool::instance();
    const size_t n = 100003;  // odd, not divisible by thread count
    std::vector<double> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<double>(i % 7) - 3.0;
    }

    std::atomic<double> partial{0.0};
    pool.parallel_for(n, 1, [&](size_t lo, size_t hi) {
        double s = 0.0;
        for (size_t i = lo; i < hi; ++i) {
            s += data[i];
        }
        partial.fetch_add(s, std::memory_order_relaxed);
    });

    const double expected = std::accumulate(data.begin(), data.end(), 0.0);
    EXPECT_NEAR(partial.load(), expected, 1e-6);
}

TEST(SpinPool, EveryItemTouchedExactlyOnce)
{
    auto &pool = SpinPool::instance();
    const size_t n = 4099;
    std::vector<std::atomic<int>> hits(n);
    for (auto &h : hits) {
        h.store(0);
    }

    pool.parallel_for(n, 1, [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
            hits[i].fetch_add(1, std::memory_order_relaxed);
        }
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hits[i].load(), 1) << "item " << i;
    }
}

TEST(SpinPool, SequentialDispatchesAreIndependent)
{
    auto &pool = SpinPool::instance();
    // Back-to-back jobs must not leak state between epochs.
    for (int round = 0; round < 50; ++round) {
        std::atomic<int> sum{0};
        pool.parallel_for(1000, 1, [&](size_t lo, size_t hi) {
            int s = 0;
            for (size_t i = lo; i < hi; ++i) {
                s += static_cast<int>(i);
            }
            sum.fetch_add(s, std::memory_order_relaxed);
        });
        const int expected = 1000 * 999 / 2;
        EXPECT_EQ(sum.load(), expected) << "round " << round;
    }
}

TEST(SpinPool, EmptyAndTinyInputs)
{
    auto &pool = SpinPool::instance();
    std::atomic<int> calls{0};
    pool.parallel_for(0, 1, [&](size_t, size_t) { calls.fetch_add(1); });
    pool.parallel_for(1, 1, [&](size_t lo, size_t hi) {
        EXPECT_EQ(lo, 0u);
        EXPECT_EQ(hi, 1u);
        calls.fetch_add(1);
    });
    EXPECT_GE(calls.load(), 1);
}

TEST(SpinPool, DispatchLatencySmoke)
{
    // Sanity: 10000 empty dispatches should complete quickly even on a
    // loaded machine (catches deadlocks/pathological spinning).
    auto &pool = SpinPool::instance();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10000; ++i) {
        pool.parallel_for(8, 1, [](size_t, size_t) {});
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    EXPECT_LT(ms, 10000) << "dispatch pathologically slow";
}
