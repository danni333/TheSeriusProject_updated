#include "stress_runner.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <sstream>
#include <numeric>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace serius {
namespace stress {

StressResult StressRunner::run_ramp(const StressConfig& cfg) {
    StressResult result;
    std::mutex mtx;
    std::atomic<bool> running{true};

    auto start_time = std::chrono::steady_clock::now();
    int current_users = 1;

    while (current_users <= cfg.ramp_max_users) {
        auto step_start = std::chrono::steady_clock::now();
        std::vector<std::thread> workers;

        for (int i = 0; i < current_users; ++i) {
            workers.emplace_back(&StressRunner::worker, this,
                                cfg.server_url, cfg.prompt, cfg.max_tokens,
                                std::ref(running), std::ref(result), std::ref(mtx));
        }

        // Wait for ramp step duration
        std::this_thread::sleep_for(std::chrono::seconds(cfg.ramp_step_seconds));

        // Record metrics
        auto step_elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - step_start).count();
        {
            std::lock_guard<std::mutex> lock(mtx);
            float tps = result.total_requests / step_elapsed;
            result.tps_timeline.push_back({step_elapsed, tps});
            result.vram_timeline.push_back({step_elapsed, get_vram_usage()});
            result.ram_timeline.push_back({step_elapsed, get_ram_usage()});
        }

        // Stop current workers
        running = false;
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }

        // Double users for next step
        current_users *= 2;
        running = true;
    }

    // Calculate final metrics
    {
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
        result.avg_tps = result.total_requests / total_elapsed;
    }

    return result;
}

StressResult StressRunner::run_soak(const StressConfig& cfg) {
    StressResult result;
    std::mutex mtx;
    std::atomic<bool> running{true};

    auto start_time = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;

    for (int i = 0; i < cfg.concurrent_users; ++i) {
        workers.emplace_back(&StressRunner::worker, this,
                            cfg.server_url, cfg.prompt, cfg.max_tokens,
                            std::ref(running), std::ref(result), std::ref(mtx));
    }

    // Run for specified duration
    auto end_time = start_time + std::chrono::seconds(cfg.duration_seconds);
    while (std::chrono::steady_clock::now() < end_time) {
        std::this_thread::sleep_for(std::chrono::seconds(60));  // Sample every minute

        auto elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - start_time).count();
        {
            std::lock_guard<std::mutex> lock(mtx);
            float tps = result.total_requests / elapsed;
            result.tps_timeline.push_back({elapsed, tps});
            result.vram_timeline.push_back({elapsed, get_vram_usage()});
            result.ram_timeline.push_back({elapsed, get_ram_usage()});
        }
    }

    running = false;
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    // Calculate final metrics
    {
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
        result.avg_tps = result.total_requests / total_elapsed;
    }

    return result;
}

StressResult StressRunner::run_spike(const StressConfig& cfg) {
    StressResult result;
    std::mutex mtx;
    std::atomic<bool> running{true};

    auto start_time = std::chrono::steady_clock::now();

    // Phase 1: Normal load
    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back(&StressRunner::worker, this,
                            cfg.server_url, cfg.prompt, cfg.max_tokens,
                            std::ref(running), std::ref(result), std::ref(mtx));
    }
    std::this_thread::sleep_for(std::chrono::seconds(30));

    // Phase 2: Spike
    running = false;
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
    workers.clear();
    running = true;

    for (int i = 0; i < cfg.spike_users; ++i) {
        workers.emplace_back(&StressRunner::worker, this,
                            cfg.server_url, cfg.prompt, cfg.max_tokens,
                            std::ref(running), std::ref(result), std::ref(mtx));
    }
    std::this_thread::sleep_for(std::chrono::seconds(cfg.spike_duration_seconds));

    // Phase 3: Recovery
    running = false;
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
    workers.clear();
    running = true;

    for (int i = 0; i < 4; ++i) {
        workers.emplace_back(&StressRunner::worker, this,
                            cfg.server_url, cfg.prompt, cfg.max_tokens,
                            std::ref(running), std::ref(result), std::ref(mtx));
    }
    std::this_thread::sleep_for(std::chrono::seconds(30));

    running = false;
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    // Calculate final metrics
    {
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
        result.avg_tps = result.total_requests / total_elapsed;
    }

    return result;
}

void StressRunner::run_chaos(const std::string& server_url) {
    // Test empty input
    httplib::Client cli(server_url);
    nlohmann::json body;
    body["model"] = "test";
    body["messages"] = nlohmann::json::array();
    body["max_tokens"] = 10;
    body["stream"] = false;

    auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    // Should return 400 error, not crash

    // Test extremely long context
    std::string long_prompt(100000, 'a');
    body["messages"] = {{{"role", "user"}, {"content", long_prompt}}};
    res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    // Should return error gracefully

    // Test malformed JSON
    res = cli.Post("/v1/chat/completions", "invalid json", "application/json");
    // Should return 400, not segfault
}

void StressRunner::worker(const std::string& url, const std::string& prompt,
                         int max_tokens, std::atomic<bool>& running,
                         StressResult& result, std::mutex& mtx) {
    httplib::Client cli(url);
    cli.set_connection_timeout(30);
    cli.set_read_timeout(30);

    while (running) {
        float latency_ms = 0.0f;
        bool success = send_request(url, prompt, max_tokens, latency_ms);

        std::lock_guard<std::mutex> lock(mtx);
        result.total_requests++;
        if (success) {
            result.latencies_ms.push_back(latency_ms);
        } else {
            result.total_errors++;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool StressRunner::send_request(const std::string& url, const std::string& prompt,
                                int max_tokens, float& latency_ms) {
    httplib::Client cli(url);
    cli.set_connection_timeout(30);
    cli.set_read_timeout(30);

    nlohmann::json body;
    body["model"] = "serius";
    body["messages"] = {{{"role", "user"}, {"content", prompt}}};
    body["max_tokens"] = max_tokens;
    body["stream"] = false;

    auto start = std::chrono::steady_clock::now();
    auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    auto end = std::chrono::steady_clock::now();

    latency_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return res && res->status == 200;
}

size_t StressRunner::get_vram_usage() {
    // Placeholder: integrate with CUDA backend for actual VRAM monitoring
    // For now, return 0
    return 0;
}

size_t StressRunner::get_ram_usage() {
    // Placeholder: integrate with OS-specific memory APIs
    // For now, return 0
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
