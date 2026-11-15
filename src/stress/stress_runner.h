#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>
#include <utility>

#include "../common/defaults.h"

namespace serius {
namespace stress {

struct StressConfig {
    int concurrent_users = 1;
    int duration_seconds = 60;
    int ramp_step_seconds = 60;
    std::string prompt = "Explain quantum computing in detail...";
    int max_tokens = defaults::kMaxNewTokens;
    std::string server_url = defaults::kServerUrl;
    std::string model_name = defaults::kModelName;
    std::string endpoint = "/v1/chat/completions";
    int http_timeout_seconds = defaults::kStressTimeoutSec;
    int request_interval_ms = 100;  // delay between requests issued by each worker
    int ramp_max_users = 128;
    int spike_users = 64;
    int spike_duration_seconds = 60;
    int spike_baseline_users = 4;
    int spike_phase_seconds = 30;   // duration of baseline and recovery phases
    int soak_sample_interval_seconds = 60;
    int chaos_max_tokens = 10;
    size_t chaos_long_prompt_chars = 100000;
};

struct StressResult {
    float avg_tps = 0.0f;
    float p50_latency_ms = 0.0f;
    float p99_latency_ms = 0.0f;
    float error_rate = 0.0f;
    size_t total_requests = 0;
    size_t total_errors = 0;
    size_t total_tokens = 0;  // sum of usage.completion_tokens across requests
    std::vector<std::pair<float, float>> tps_timeline;  // (timestamp, tps)
    std::vector<std::pair<float, size_t>> vram_timeline; // (timestamp, bytes)
    std::vector<std::pair<float, size_t>> ram_timeline;  // (timestamp, bytes)
    std::vector<float> latencies_ms;
};

struct ChaosResult {
    int checks_passed = 0;
    int checks_failed = 0;
    bool server_alive = false;  // /health still responds after the probes
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

    /// Run chaos test: malformed inputs and error scenarios.
    /// Each probe is checked: invalid inputs must yield a client error (4xx),
    /// and the server must still answer /health afterwards.
    ChaosResult run_chaos(const StressConfig& cfg);

    /// Calculate percentile from latency values
    float calculate_percentile(std::vector<float>& values, float percentile);

private:
    /// Compute final metrics (percentiles, error rate, avg TPS) under the result mutex.
    void finalize_metrics(StressResult& result, std::mutex& mtx,
                          std::chrono::steady_clock::time_point start_time);

    /// Sample the SERVER's RAM/VRAM via GET /health (the stress process's own
    /// memory is irrelevant for leak detection). Returns {ram_bytes, vram_bytes}.
    std::pair<size_t, size_t> sample_server_memory(const std::string &server_url);
};

}  // namespace stress
}  // namespace serius
