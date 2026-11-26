#include "stress_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <thread>

#include "worker.h"
#if defined(_MSC_VER)
// cpp-httplib 0.67 trips C4702 (unreachable code) inside its own headers;
// MSVC's -external:W0 does not suppress it under /WX.
#pragma warning(push)
#pragma warning(disable : 4702)
#endif
#include <httplib.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#include <nlohmann/json.hpp>

namespace serius {
namespace stress {

namespace {

/// Base URL used for /health memory sampling: the dedicated ops port when
/// configured, otherwise the main server URL.
/// httplib may resolve "localhost" to ::1 (IPv6) first while the server
/// binds IPv4-only (0.0.0.0); the connect then fails with Error::Connection
/// and every request errors out. Map localhost to 127.0.0.1 so stress
/// clients always reach an IPv4 listener.
std::string canonical_url(const std::string &url)
{
    constexpr const char *kFrom = "://localhost";
    constexpr const char *kTo = "://127.0.0.1";
    const size_t pos = url.find(kFrom);
    if (pos == std::string::npos) {
        return url;
    }
    std::string out = url;
    out.replace(pos, std::strlen(kFrom), kTo);
    return out;
}

std::string health_base(const StressConfig &cfg)
{
    return canonical_url(cfg.health_url.empty() ? cfg.server_url : cfg.health_url);
}

/// Spawn `count` worker threads running StressWorker with the given config.
void spawn_workers(int count, const StressConfig &cfg, std::atomic<bool> &running, StressResult &result,
                   std::mutex &mtx, std::vector<std::thread> &workers)
{
    workers.reserve(workers.size() + static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        workers.emplace_back([&cfg, &running, &result, &mtx]() {
            StressWorker worker(canonical_url(cfg.server_url), cfg.prompt, cfg.max_tokens, cfg.model_name,
                                cfg.endpoint, cfg.http_timeout_seconds, cfg.request_interval_ms);
            worker.run(running, result, mtx);
        });
    }
}

void join_workers(std::vector<std::thread> &workers)
{
    for (auto &w : workers) {
        if (w.joinable()) w.join();
    }
    workers.clear();
}

}  // namespace

StressResult StressRunner::run_ramp(const StressConfig &cfg)
{
    StressResult result;
    std::mutex mtx;
    std::atomic<bool> running{true};

    auto start_time = std::chrono::steady_clock::now();
    int current_users = 1;

    while (current_users <= cfg.ramp_max_users) {
        auto step_start = std::chrono::steady_clock::now();
        std::vector<std::thread> workers;

        size_t tokens_before = 0;
        {
            std::lock_guard<std::mutex> lock(mtx);
            tokens_before = result.total_tokens;
        }

        spawn_workers(current_users, cfg, running, result, mtx, workers);

        // Wait for ramp step duration
        std::this_thread::sleep_for(std::chrono::seconds(cfg.ramp_step_seconds));

        // Record metrics for this step only (delta since step start)
        auto step_elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - step_start).count();
        const auto [ram_bytes, vram_bytes] = sample_server_memory(health_base(cfg));
        {
            std::lock_guard<std::mutex> lock(mtx);
            float step_tokens = static_cast<float>(result.total_tokens - tokens_before);
            float tps = step_elapsed > 0.0f ? step_tokens / step_elapsed : 0.0f;
            result.tps_timeline.push_back({step_elapsed, tps});
            result.vram_timeline.push_back({step_elapsed, vram_bytes});
            result.ram_timeline.push_back({step_elapsed, ram_bytes});
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

StressResult StressRunner::run_soak(const StressConfig &cfg)
{
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

        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
        const auto [ram_bytes, vram_bytes] = sample_server_memory(health_base(cfg));
        {
            std::lock_guard<std::mutex> lock(mtx);
            float tps = elapsed > 0.0f ? static_cast<float>(result.total_tokens) / elapsed : 0.0f;
            result.tps_timeline.push_back({elapsed, tps});
            result.vram_timeline.push_back({elapsed, vram_bytes});
            result.ram_timeline.push_back({elapsed, ram_bytes});
        }
    }

    running = false;
    join_workers(workers);

    finalize_metrics(result, mtx, start_time);
    return result;
}

StressResult StressRunner::run_spike(const StressConfig &cfg)
{
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

ChaosResult StressRunner::run_chaos(const StressConfig &cfg)
{
    ChaosResult result;
    httplib::Client cli(cfg.server_url);
    cli.set_connection_timeout(cfg.http_timeout_seconds);
    cli.set_read_timeout(cfg.http_timeout_seconds);

    auto probe = [&](const char *name, const std::string &body, bool expect_client_error) {
        auto res = cli.Post(cfg.endpoint, body, "application/json");
        const bool got_response = static_cast<bool>(res);
        // expect_client_error: require a 4xx. Otherwise any HTTP response
        // below 500 counts as graceful handling — a 500 is a server failure,
        // not a pass.
        const bool ok = got_response && (expect_client_error ? (res->status >= 400 && res->status < 500)
                                                            : (res->status < 500));
        if (ok) {
            ++result.checks_passed;
        } else {
            ++result.checks_failed;
        }
        (void)name;
    };

    // Empty messages array -> client error, not crash.
    {
        nlohmann::json body;
        body["model"] = cfg.model_name;
        body["messages"] = nlohmann::json::array();
        body["max_tokens"] = cfg.chaos_max_tokens;
        body["stream"] = false;
        probe("empty messages", body.dump(), true);
    }

    // Extremely long context -> client error or graceful handling.
    {
        nlohmann::json body;
        body["model"] = cfg.model_name;
        body["messages"] = {{{"role", "user"}, {"content", std::string(cfg.chaos_long_prompt_chars, 'a')}}};
        body["max_tokens"] = cfg.chaos_max_tokens;
        body["stream"] = false;
        probe("long context", body.dump(), false);
    }

    // Malformed JSON -> client error, not segfault.
    probe("malformed JSON", "invalid json", true);

    // Server must still be alive afterwards.
    if (auto res = cli.Get("/health"); res && res->status == 200) {
        result.server_alive = true;
    }
    return result;
}

void StressRunner::finalize_metrics(StressResult &result, std::mutex &mtx,
                                    std::chrono::steady_clock::time_point start_time)
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
    auto total_elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
    result.avg_tps = total_elapsed > 0.0f ? static_cast<float>(result.total_tokens) / total_elapsed : 0.0f;
}

std::pair<size_t, size_t> StressRunner::sample_server_memory(const std::string &server_url)
{
    httplib::Client cli(server_url);
    cli.set_connection_timeout(defaults::kStressChaosTimeoutSec);
    cli.set_read_timeout(defaults::kStressChaosTimeoutSec);
    if (auto res = cli.Get("/health"); res && res->status == 200) {
        try {
            const auto j = nlohmann::json::parse(res->body);
            return {j.value("ram_bytes", size_t{0}), j.value("vram_bytes", size_t{0})};
        }
        catch (...) {
        }
    }
    return {0, 0};
}

float StressRunner::calculate_percentile(std::vector<float> &values, float percentile)
{
    if (values.empty()) return 0.0f;

    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>(std::ceil(percentile / 100.0f * values.size())) - 1;
    idx = std::min(idx, values.size() - 1);
    return values[idx];
}

}  // namespace stress
}  // namespace serius
