#pragma once
/// @file spin_pool.h
/// @brief Persistent-thread pool with spin-wait dispatch (ggml-style).
///
/// Fase 73e — replaces per-call `#pragma omp parallel for` wake-ups in hot
/// decode paths. Workers spin on an atomic epoch counter instead of being
/// scheduled by the OS each parallel region.
///
/// Usage:
///   auto &pool = SpinPool::instance();
///   pool.parallel_for(n_items, grain, [&](size_t lo, size_t hi) { ... });
///
/// The calling thread participates as worker 0. `fn` is invoked once per
/// chunk with a contiguous [lo, hi) item range; chunks are split evenly.
/// Thread count: SERIUS_SPIN_THREADS env, else min(hardware_concurrency, 8).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace serius {

class SpinPool
{
   public:
    static SpinPool &instance()
    {
        static SpinPool pool;
        return pool;
    }

    size_t num_workers() const { return workers_.size() + 1; }  // + caller

    /// Parallel-for over [0, n_items) split into ~num_workers contiguous
    /// chunks; `fn(lo, hi)` runs the chunk body. grain is reserved for
    /// future dynamic scheduling (currently unused: static split).
    void parallel_for(size_t n_items, size_t grain, const std::function<void(size_t, size_t)> &fn);

    /// SERIUS_SPINPOOL=1 enables the pool in hot paths (matvec_q,
    /// attention). Default off until the end-to-end A/B proves a win.
    static bool enabled()
    {
        static const bool on = [] {
#if defined(_MSC_VER)
            char buf[8];
            size_t len = 0;
            return getenv_s(&len, buf, sizeof(buf), "SERIUS_SPINPOOL") == 0 && len > 0 &&
                   buf[0] == '1';
#else
            const char *e = std::getenv("SERIUS_SPINPOOL");
            return e != nullptr && e[0] == '1';
#endif
        }();
        return on;
    }

    /// Hint that a burst of jobs is starting/ending (idle policy hook,
    /// implemented in Fase D). Currently a no-op.
    void begin_burst() {}
    void end_burst() {}

   private:
    SpinPool();
    ~SpinPool();
    SpinPool(const SpinPool &) = delete;
    SpinPool &operator=(const SpinPool &) = delete;

    struct Shared
    {
        std::atomic<uint64_t> epoch{0};
        std::atomic<int> remaining{0};
        std::atomic<int> ready{0};  // workers that captured their baseline epoch
        std::function<void(int)> job;  // chunk index -> body
    };

    void worker_loop(int idx);

    Shared shared_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};
    std::mutex dispatch_mtx_;
    size_t nthreads_ = 1;
};

}  // namespace serius
