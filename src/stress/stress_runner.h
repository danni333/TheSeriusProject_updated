#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>

namespace serius {
namespace stress {

struct StressConfig {
    int concurrent_users = 1;
    int duration_seconds = 60;
    int ramp_step_seconds = 60;
    std::string prompt = "Explain quantum computing in detail...";
    int max_tokens = 256;
    std::string server_url = "http://localhost:8080";
    int ramp_max_users = 128;
    int spike_users = 64;
    int spike_duration_seconds = 60;
    int request_interval_ms = 5000;  // 5 seconds between requests for soak test
};

struct StressResult {
    float avg_tps = 0.0f;
    float p50_latency_ms = 0.0f;
    float p99_latency_ms = 0.0f;
    float error_rate = 0.0f;
    size_t total_requests = 0;
    size_t total_errors = 0;
    std::vector<std::pair<float, float>> tps_timeline;  // (timestamp, tps)
    std::vector<std::pair<float, size_t>> vram_timeline; // (timestamp, bytes)
    std::vector<std::pair<float, size_t>> ram_timeline;  // (timestamp, bytes)
    std::vector<float> latencies_ms;
};

class StressRunner {
public:
    StressRunner() = default;
    ~StressRunner() = default;

    /// Run ramping load test: increases users gradually
    StressResult run_ramp(const StressConfig& cfg);

    /// Run soak test: constant load for extended duration
    StressResult run_soak(const StressConfig& cfg);

    /// Run spike test: sudden traffic spike
    StressResult run_spike(const StressConfig& cfg);

    /// Run chaos test: malformed inputs and error scenarios
    void run_chaos(const std::string& server_url);

private:
    /// Worker thread that simulates a single user
    void worker(const std::string& url, const std::string& prompt,
                int max_tokens, std::atomic<bool>& running,
                StressResult& result, std::mutex& mtx);

    /// Send HTTP request to server
    bool send_request(const std::string& url, const std::string& prompt,
                      int max_tokens, float& latency_ms);

    /// Get current VRAM usage (in bytes)
    size_t get_vram_usage();

    /// Get current RAM usage (in bytes)
    size_t get_ram_usage();

    /// Calculate percentile from latency values
    float calculate_percentile(std::vector<float>& values, float percentile);
};

}  // namespace stress
}  // namespace serius
