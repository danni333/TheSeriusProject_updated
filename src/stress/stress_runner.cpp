#include "stress_runner.h"
#include "worker.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <sstream>
#include <numeric>
#include <cstdlib>
#include <httplib.h>
#include <nlohmann/json.hpp>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#elif defined(__linux__)
#include <fstream>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

#ifdef SERIUS_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace serius {
namespace stress {

namespace {

/// Spawn `count` worker threads running StressWorker with the given config.
void spawn_workers(int count, const StressConfig& cfg,
                   std::atomic<bool>& running, StressResult& result,
                   std::mutex& mtx, std::vector<std::thread>& workers) {
    workers.reserve(workers.size() + static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        workers.emplace_back([&cfg, &running, &result, &mtx]() {
            StressWorker worker(cfg.server_url, cfg.prompt, cfg.max_tokens,
                                cfg.model_name, cfg.endpoint,
                                cfg.http_timeout_seconds, cfg.request_interval_ms);
            worker.run(running, result, mtx);
        });
    }
}

void join_workers(std::vector<std::thread>& workers) {
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
    workers.clear();
}

}  // namespace

StressResult StressRunner::run_ramp(const StressConfig& cfg) {
    StressResult result;
    std::mutex mtx;
    std::atomic<bool> running{true};

    auto start_time = std::chrono::steady_clock::now();
    int current_users = 1;

    while (current_users <= cfg.ramp_max_users) {
        auto step_start = std::chrono::steady_clock::now();
        std::vector<std::thread> workers;

        size_t requests_before = 0;
        {
            std::lock_guard<std::mutex> lock(mtx);
            requests_before = result.total_requests;
        }

        spawn_workers(current_users, cfg, running, result, mtx, workers);

        // Wait for ramp step duration
        std::this_thread::sleep_for(std::chrono::seconds(cfg.ramp_step_seconds));

        // Record metrics for this step only (delta since step start)
        auto step_elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - step_start).count();
        {
            std::lock_guard<std::mutex> lock(mtx);
            float step_requests = static_cast<float>(result.total_requests - requests_before);
            float tps = step_elapsed > 0.0f ? step_requests / step_elapsed : 0.0f;
            result.tps_timeline.push_back({step_elapsed, tps});
            result.vram_timeline.push_back({step_elapsed, get_vram_usage()});
            result.ram_timeline.push_back({step_elapsed, get_ram_usage()});
        }

        // Stop current workers
        running = false;
        join_workers(workers);

        // Double users for next step
        current_users *= 2;
        running = true;
    }

    finalize_metrics(result, mtx, start_time);
    return result;
}

StressResult StressRunner::run_soak(const StressConfig& cfg) {
    StressResult result;
    std::mutex mtx;
    std::atomic<bool> running{true};

    auto start_time = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;

    spawn_workers(cfg.concurrent_users, cfg, running, result, mtx, workers);

    // Run for specified duration, sampling memory periodically
    auto end_time = start_time + std::chrono::seconds(cfg.duration_seconds);
    while (std::chrono::steady_clock::now() < end_time) {
        auto remaining = std::chrono::duration_cast<std::chrono::seconds>(end_time - std::chrono::steady_clock::now());
        std::this_thread::sleep_for(std::chrono::seconds(
            std::min<int64_t>(cfg.soak_sample_interval_seconds, std::max<int64_t>(remaining.count(), 1))));

        auto elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - start_time).count();
        {
            std::lock_guard<std::mutex> lock(mtx);
            float tps = elapsed > 0.0f
                ? static_cast<float>(result.total_requests) / elapsed
                : 0.0f;
            result.tps_timeline.push_back({elapsed, tps});
            result.vram_timeline.push_back({elapsed, get_vram_usage()});
            result.ram_timeline.push_back({elapsed, get_ram_usage()});
        }
    }

    running = false;
    join_workers(workers);

    finalize_metrics(result, mtx, start_time);
    return result;
}

StressResult StressRunner::run_spike(const StressConfig& cfg) {
    StressResult result;
    std::mutex mtx;
    std::atomic<bool> running{true};

    auto start_time = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;

    // Phase 1: baseline load
    spawn_workers(cfg.spike_baseline_users, cfg, running, result, mtx, workers);
    std::this_thread::sleep_for(std::chrono::seconds(cfg.spike_phase_seconds));

    // Phase 2: spike
    running = false;
    join_workers(workers);
    running = true;

    spawn_workers(cfg.spike_users, cfg, running, result, mtx, workers);
    std::this_thread::sleep_for(std::chrono::seconds(cfg.spike_duration_seconds));

    // Phase 3: recovery to baseline
    running = false;
    join_workers(workers);
    running = true;

    spawn_workers(cfg.spike_baseline_users, cfg, running, result, mtx, workers);
    std::this_thread::sleep_for(std::chrono::seconds(cfg.spike_phase_seconds));

    running = false;
    join_workers(workers);

    finalize_metrics(result, mtx, start_time);
    return result;
}

void StressRunner::run_chaos(const StressConfig& cfg) {
    httplib::Client cli(cfg.server_url);
    cli.set_connection_timeout(cfg.http_timeout_seconds);
    cli.set_read_timeout(cfg.http_timeout_seconds);

    // Test empty input
    nlohmann::json body;
    body["model"] = cfg.model_name;
    body["messages"] = nlohmann::json::array();
    body["max_tokens"] = cfg.chaos_max_tokens;
    body["stream"] = false;

    cli.Post(cfg.endpoint, body.dump(), "application/json");
    // Should return 400 error, not crash

    // Test extremely long context
    std::string long_prompt(cfg.chaos_long_prompt_chars, 'a');
    body["messages"] = {{{"role", "user"}, {"content", long_prompt}}};
    cli.Post(cfg.endpoint, body.dump(), "application/json");
    // Should return error gracefully

    // Test malformed JSON
    cli.Post(cfg.endpoint, "invalid json", "application/json");
    // Should return 400, not segfault
}

void StressRunner::finalize_metrics(StressResult& result, std::mutex& mtx,
                                    std::chrono::steady_clock::time_point start_time) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!result.latencies_ms.empty()) {
        std::sort(result.latencies_ms.begin(), result.latencies_ms.end());
        result.p50_latency_ms = calculate_percentile(result.latencies_ms, 50.0f);
        result.p99_latency_ms = calculate_percentile(result.latencies_ms, 99.0f);
    }
    if (result.total_requests > 0) {
        result.error_rate = static_cast<float>(result.total_errors) / result.total_requests;
    }
    auto total_elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - start_time).count();
    result.avg_tps = total_elapsed > 0.0f
        ? static_cast<float>(result.total_requests) / total_elapsed
        : 0.0f;
}

size_t StressRunner::get_vram_usage() {
#ifdef SERIUS_USE_CUDA
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        return total_bytes > free_bytes ? total_bytes - free_bytes : 0;
    }
#endif
    return 0;
}

size_t StressRunner::get_ram_usage() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                             sizeof(pmc))) {
        return static_cast<size_t>(pmc.WorkingSetSize);
    }
#elif defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            // Format: "VmRSS:   12345 kB"
            return std::strtoull(line.c_str() + 6, nullptr, 10) * 1024;
        }
    }
#elif defined(__APPLE__)
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<size_t>(info.resident_size);
    }
#endif
    return 0;
}

float StressRunner::calculate_percentile(std::vector<float>& values, float percentile) {
    if (values.empty()) return 0.0f;

    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>(std::ceil(percentile / 100.0f * values.size())) - 1;
    idx = std::min(idx, values.size() - 1);
    return values[idx];
}

}  // namespace stress
}  // namespace serius
