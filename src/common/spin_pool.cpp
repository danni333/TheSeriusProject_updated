#include "common/spin_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#ifdef _MSC_VER
#include <immintrin.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef __unix__
#include <sched.h>
#endif

namespace serius {

namespace {

inline int env_int(const char *name, int fallback)
{
#ifdef _MSC_VER
    char buf[16];
    size_t len = 0;
    if (getenv_s(&len, buf, sizeof(buf), name) == 0 && len > 0) {
        return std::atoi(buf);
    }
#else
    const char *e = std::getenv(name);
    if (e != nullptr) {
        return std::atoi(e);
    }
#endif
    return fallback;
}

inline void spin_pause()
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

inline void relax()
{
#ifdef _WIN32
    Sleep(0);
#elif defined(__unix__)
    sched_yield();
#else
    std::this_thread::yield();
#endif
}

}  // namespace

SpinPool::SpinPool()
{
    int n = env_int("SERIUS_SPIN_THREADS", 0);
    if (n <= 0) {
        n = env_int("OMP_NUM_THREADS", 0);  // respect the user's OMP choice
    }
    if (n <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        n = hc > 0 ? static_cast<int>(hc) : 4;
        if (n > 8) {
            n = 8;  // HT siblings share ports; memory-bound work prefers P-cores
        }
    }
    nthreads_ = static_cast<size_t>(n);

    workers_.reserve(nthreads_ - 1);
    for (size_t i = 1; i < nthreads_; ++i) {
        workers_.emplace_back([this, i] { worker_loop(static_cast<int>(i)); });
    }
    // Ready barrier: a worker that reads `seen` after a dispatch already
    // bumped epoch would skip that job and hang the caller on `remaining`.
    // Block until every worker captured its baseline epoch.
    while (shared_.ready.load(std::memory_order_acquire) !=
           static_cast<int>(workers_.size())) {
        spin_pause();
    }
}

SpinPool::~SpinPool()
{
    // Do NOT bump epoch here: a worker seeing a new epoch would run
    // shared_.job, whose captured references may already be dangling.
    // Workers poll stop_ every spin iteration and exit on their own.
    stop_.store(true, std::memory_order_release);
    for (auto &t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void SpinPool::worker_loop(int idx)
{
    uint64_t seen = shared_.epoch.load(std::memory_order_acquire);
    shared_.ready.fetch_add(1, std::memory_order_release);
    long idle_spins = 0;
    while (!stop_.load(std::memory_order_acquire)) {
        const uint64_t e = shared_.epoch.load(std::memory_order_acquire);
        if (e != seen) {
            seen = e;
            if (stop_.load(std::memory_order_acquire)) {
                break;
            }
            // Copy the std::function locally: the caller may overwrite
            // shared_.job only after remaining hits 0, which cannot
            // happen before this worker decrements it.
            auto job = shared_.job;
            job(idx);
            shared_.remaining.fetch_sub(1, std::memory_order_release);
            idle_spins = 0;
        } else {
            // Spin briefly, then yield to avoid burning a core forever.
            if (++idle_spins > 200000) {
                relax();
                idle_spins = 0;
            } else {
                spin_pause();
            }
        }
    }
}

void SpinPool::parallel_for(size_t n_items, size_t /*grain*/,
                            const std::function<void(size_t, size_t)> &fn)
{
    // Serialize dispatchers: shared_.job/epoch are single-slot. Decode is
    // single-threaded, but the HTTP server can run concurrent CPU requests.
    std::lock_guard<std::mutex> dispatch_lock(dispatch_mtx_);
    if (nthreads_ <= 1 || n_items == 0) {
        if (n_items > 0) {
            fn(0, n_items);
        }
        return;
    }

    const int T = static_cast<int>(nthreads_);
    // Static contiguous split, ggml-style.
    shared_.job = [fn, n_items, T](int chunk) {
        const size_t base = n_items / static_cast<size_t>(T);
        const size_t rem = n_items % static_cast<size_t>(T);
        const size_t lo = static_cast<size_t>(chunk) * base +
                          std::min(static_cast<size_t>(chunk), rem);
        const size_t hi = lo + base + (static_cast<size_t>(chunk) < rem ? 1 : 0);
        if (lo < hi) {
            fn(lo, hi);
        }
    };
    shared_.remaining.store(T - 1, std::memory_order_relaxed);
    // Release: workers must see job + remaining before epoch.
    shared_.epoch.fetch_add(1, std::memory_order_release);

    // Caller runs chunk 0.
    shared_.job(0);

    // Wait for workers. Spin with pause; yield if it takes long.
    long spins = 0;
    while (shared_.remaining.load(std::memory_order_acquire) != 0) {
        if (++spins > 2000000) {
            relax();
            spins = 0;
        } else {
            spin_pause();
        }
    }
}

}  // namespace serius
