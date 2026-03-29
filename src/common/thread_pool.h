#pragma once
/// @file thread_pool.h
/// @brief Thread pool with work-stealing for CPU parallelism.
///
/// Serius Phase 1 (T1.01) — Provides a global thread pool and
/// parallel_for / parallel_for_chunked helpers used by BLAS
/// and other CPU-intensive workloads.
///
/// Usage:
///   serius::parallel_for(0, M, [&](size_t row) { ... });

#include <atomic>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace serius {

/// A simple, MSVC-compatible thread pool.
class ThreadPool
{
   public:
    /// Get the global singleton.  Lazily initialized on first call.
    static ThreadPool &instance()
    {
        static ThreadPool pool;
        return pool;
    }

    /// Number of worker threads (== hardware_concurrency by default).
    size_t num_threads() const { return workers_.size(); }

    /// Submit a void() callable and get a future to wait on.
    std::future<void> submit(std::function<void()> task_fn)
    {
        auto promise = std::make_shared<std::promise<void>>();
        std::future<void> fut = promise->get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            task_queue_.push([task_fn = std::move(task_fn), promise]() {
                try {
                    task_fn();
                    promise->set_value();
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            });
        }
        cv_.notify_one();

        return fut;
    }

    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // Non-copyable, non-movable.
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

   private:
    ThreadPool()
    {
        unsigned num_hw = std::thread::hardware_concurrency();
        if (num_hw == 0) {
            num_hw = 4;  // fallback
        }

        workers_.reserve(num_hw);
        for (unsigned idx = 0; idx < num_hw; ++idx) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    void worker_loop()
    {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mtx_);
                cv_.wait(lock, [this]() { return stop_ || !task_queue_.empty(); });
                if (stop_ && task_queue_.empty()) {
                    return;
                }
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mtx_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// ──────────────────────────────────────────────────────────────
//  parallel_for — the main API for BLAS and other hot loops
// ──────────────────────────────────────────────────────────────

/// Execute `func(i)` for i in [begin, end) distributed across the pool.
/// If the range is small the work runs inline (no thread overhead).
template <typename Func>
void parallel_for(size_t begin, size_t end, Func &&func)
{
    if (begin >= end) {
        return;
    }

    const size_t total = end - begin;

    // For tiny ranges don't bother with the pool
    constexpr size_t MIN_PARALLEL_SIZE = 4;
    auto &pool = ThreadPool::instance();
    const size_t num_workers = pool.num_threads();

    if (total <= MIN_PARALLEL_SIZE || num_workers <= 1) {
        for (size_t idx = begin; idx < end; ++idx) {
            func(idx);
        }
        return;
    }

    // Split into roughly equal chunks
    const size_t n_chunks = std::min(total, num_workers);
    const size_t chunk_size = (total + n_chunks - 1) / n_chunks;

    std::vector<std::future<void>> futures;
    futures.reserve(n_chunks);

    for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
        size_t chunk_lo = begin + (chunk * chunk_size);
        size_t chunk_hi = std::min(chunk_lo + chunk_size, end);
        if (chunk_lo >= chunk_hi) {
            break;
        }

        futures.push_back(pool.submit([chunk_lo, chunk_hi, &func]() {
            for (size_t idx = chunk_lo; idx < chunk_hi; ++idx) {
                func(idx);
            }
        }));
    }

    // Wait for all chunks
    for (auto &fut : futures) {
        fut.get();
    }
}

/// Same as parallel_for but passes (chunk_begin, chunk_end) to func,
/// allowing the callee to process a contiguous range (better for BLAS).
template <typename Func>
void parallel_for_chunked(size_t begin, size_t end, Func &&func)
{
    if (begin >= end) {
        return;
    }

    const size_t total = end - begin;
    auto &pool = ThreadPool::instance();
    const size_t num_workers = pool.num_threads();

    constexpr size_t MIN_PARALLEL_SIZE = 4;
    if (total <= MIN_PARALLEL_SIZE || num_workers <= 1) {
        func(begin, end);
        return;
    }

    const size_t n_chunks = std::min(total, num_workers);
    const size_t chunk_size = (total + n_chunks - 1) / n_chunks;

    std::vector<std::future<void>> futures;
    futures.reserve(n_chunks);

    for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
        size_t chunk_lo = begin + (chunk * chunk_size);
        size_t chunk_hi = std::min(chunk_lo + chunk_size, end);
        if (chunk_lo >= chunk_hi) {
            break;
        }

        futures.push_back(pool.submit([chunk_lo, chunk_hi, &func]() { func(chunk_lo, chunk_hi); }));
    }

    for (auto &fut : futures) {
        fut.get();
    }
}

}  // namespace serius
