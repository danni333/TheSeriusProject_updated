#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <httplib.h>
#include <atomic>
#include <thread>
#include "stress/stress_runner.h"
#include "stress/report.h"
#include "stress/leak_detector.h"

namespace serius {
namespace stress {

namespace {

/// Minimal mock of the Serius HTTP server for stress-runner tests.
/// Returns 200 with a chat-completion body for valid requests,
/// 400 for malformed JSON or empty messages.
class MockStressServer {
public:
    MockStressServer() {
        server_.Post("/v1/chat/completions",
                     [this](const httplib::Request& req, httplib::Response& res) {
            request_count_.fetch_add(1);
            nlohmann::json body;
            try {
                body = nlohmann::json::parse(req.body);
            } catch (...) {
                res.status = 400;
                return;
            }
            auto it = body.find("messages");
            if (it == body.end() || !it->is_array() || it->empty()) {
                res.status = 400;
                return;
            }
            res.set_content(
                R"({"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})",
                "application/json");
        });
    }

    bool start() {
        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ < 0) return false;
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        return true;
    }

    void stop() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
    int request_count() const { return request_count_.load(); }

private:
    httplib::Server server_;
    std::thread thread_;
    int port_ = -1;
    std::atomic<int> request_count_{0};
};

StressConfig test_config(const std::string& url) {
    StressConfig cfg;
    cfg.server_url = url;
    cfg.request_interval_ms = 5;
    cfg.http_timeout_seconds = 5;
    cfg.max_tokens = 4;
    return cfg;
}

}  // namespace

TEST(StressTest, StressConfigDefaults) {
    StressConfig config;
    EXPECT_EQ(config.concurrent_users, 1);
    EXPECT_EQ(config.duration_seconds, 60);
    EXPECT_EQ(config.ramp_step_seconds, 60);
    EXPECT_EQ(config.max_tokens, 256);
    EXPECT_EQ(config.server_url, "http://localhost:8080");
    EXPECT_EQ(config.ramp_max_users, 128);
}

TEST(StressTest, StressResultDefaults) {
    StressResult result;
    EXPECT_EQ(result.avg_tps, 0.0f);
    EXPECT_EQ(result.p50_latency_ms, 0.0f);
    EXPECT_EQ(result.p99_latency_ms, 0.0f);
    EXPECT_EQ(result.error_rate, 0.0f);
    EXPECT_EQ(result.total_requests, 0);
    EXPECT_EQ(result.total_errors, 0);
    EXPECT_TRUE(result.tps_timeline.empty());
    EXPECT_TRUE(result.vram_timeline.empty());
    EXPECT_TRUE(result.ram_timeline.empty());
    EXPECT_TRUE(result.latencies_ms.empty());
}

TEST(StressTest, CalculatePercentileEmpty) {
    StressRunner runner;
    std::vector<float> values;
    float p50 = runner.calculate_percentile(values, 50.0f);
    EXPECT_EQ(p50, 0.0f);
}

TEST(StressTest, CalculatePercentileSingle) {
    StressRunner runner;
    std::vector<float> values = {100.0f};
    float p50 = runner.calculate_percentile(values, 50.0f);
    EXPECT_FLOAT_EQ(p50, 100.0f);
}

TEST(StressTest, CalculatePercentileMultiple) {
    StressRunner runner;
    std::vector<float> values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    float p50 = runner.calculate_percentile(values, 50.0f);
    EXPECT_FLOAT_EQ(p50, 30.0f);

    float p99 = runner.calculate_percentile(values, 99.0f);
    EXPECT_FLOAT_EQ(p99, 50.0f);
}

TEST(StressTest, GenerateReportASCII) {
    StressResult result;
    result.avg_tps = 100.0f;
    result.p50_latency_ms = 50.0f;
    result.p99_latency_ms = 100.0f;
    result.error_rate = 0.01f;
    result.total_requests = 1000;
    result.total_errors = 10;

    StressConfig config;
    config.server_url = "http://localhost:8080";
    config.concurrent_users = 8;
    config.duration_seconds = 60;

    std::string report = generate_report(result, config, ReportFormat::ASCII);

    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("STRESS TEST REPORT"), std::string::npos);
    EXPECT_NE(report.find("100.0"), std::string::npos);
    EXPECT_NE(report.find("http://localhost:8080"), std::string::npos);
}

TEST(StressTest, GenerateReportJSON) {
    StressResult result;
    result.avg_tps = 100.0f;
    result.p50_latency_ms = 50.0f;
    result.p99_latency_ms = 100.0f;
    result.error_rate = 0.01f;
    result.total_requests = 1000;
    result.total_errors = 10;

    StressConfig config;
    config.server_url = "http://localhost:8080";
    config.concurrent_users = 8;

    std::string report = generate_report(result, config, ReportFormat::JSON);

    EXPECT_FALSE(report.empty());

    nlohmann::json j = nlohmann::json::parse(report);
    EXPECT_TRUE(j.contains("config"));
    EXPECT_TRUE(j.contains("result"));
    EXPECT_FLOAT_EQ(j["result"]["avg_tps"], 100.0f);
    EXPECT_EQ(j["result"]["total_requests"], 1000);
}

TEST(LeakDetectorTest, LeakAnalysisDefaults) {
    LeakAnalysis analysis;
    EXPECT_FALSE(analysis.vram_leak_detected);
    EXPECT_FALSE(analysis.ram_leak_detected);
    EXPECT_EQ(analysis.vram_drift_bytes, 0);
    EXPECT_EQ(analysis.ram_drift_bytes, 0);
    EXPECT_FLOAT_EQ(analysis.vram_drift_mb_per_hour, 0.0f);
    EXPECT_FLOAT_EQ(analysis.ram_drift_mb_per_hour, 0.0f);
}

TEST(LeakDetectorTest, AnalyzeInsufficientData) {
    LeakDetector detector;
    std::vector<MemorySample> samples = {{0.0f, 1024, 2048}};
    auto analysis = detector.analyze(samples, 1.0f);
    EXPECT_FALSE(analysis.vram_leak_detected);
    EXPECT_FALSE(analysis.ram_leak_detected);
    EXPECT_NE(analysis.summary.find("Insufficient data"), std::string::npos);
}

TEST(LeakDetectorTest, AnalyzeNoLeak) {
    LeakDetector detector;
    std::vector<MemorySample> samples = {
        {0.0f, 1024 * 1024, 2048 * 1024},
        {1.0f, 1024 * 1024, 2048 * 1024},
        {2.0f, 1024 * 1024, 2048 * 1024}
    };
    auto analysis = detector.analyze(samples, 2.0f);
    EXPECT_FALSE(analysis.vram_leak_detected);
    EXPECT_FALSE(analysis.ram_leak_detected);
    EXPECT_EQ(analysis.vram_drift_bytes, 0);
    EXPECT_EQ(analysis.ram_drift_bytes, 0);
}

TEST(LeakDetectorTest, AnalyzeVRAMLeak) {
    LeakDetector detector;
    std::vector<MemorySample> samples = {
        {0.0f, 100 * 1024 * 1024, 200 * 1024 * 1024},
        {1.0f, 150 * 1024 * 1024, 200 * 1024 * 1024},
        {2.0f, 200 * 1024 * 1024, 200 * 1024 * 1024}
    };
    auto analysis = detector.analyze(samples, 2.0f, 50 * 1024 * 1024);
    EXPECT_TRUE(analysis.vram_leak_detected);
    EXPECT_FALSE(analysis.ram_leak_detected);
    EXPECT_GT(analysis.vram_drift_bytes, 0);
}

TEST(LeakDetectorTest, AnalyzeRAMLeak) {
    LeakDetector detector;
    std::vector<MemorySample> samples = {
        {0.0f, 100 * 1024 * 1024, 200 * 1024 * 1024},
        {1.0f, 100 * 1024 * 1024, 250 * 1024 * 1024},
        {2.0f, 100 * 1024 * 1024, 300 * 1024 * 1024}
    };
    auto analysis = detector.analyze(samples, 2.0f, 50 * 1024 * 1024);
    EXPECT_FALSE(analysis.vram_leak_detected);
    EXPECT_TRUE(analysis.ram_leak_detected);
    EXPECT_GT(analysis.ram_drift_bytes, 0);
}

TEST(LeakDetectorTest, FormatSummary) {
    LeakDetector detector;
    LeakAnalysis analysis;
    analysis.vram_drift_bytes = 50 * 1024 * 1024;
    analysis.ram_drift_bytes = 25 * 1024 * 1024;
    analysis.vram_leak_detected = true;
    analysis.ram_leak_detected = false;
    analysis.vram_drift_mb_per_hour = 25.0f;
    analysis.ram_drift_mb_per_hour = 12.5f;

    std::string summary = detector.format_summary(analysis);
    EXPECT_FALSE(summary.empty());
    EXPECT_NE(summary.find("Memory Leak Analysis"), std::string::npos);
    EXPECT_NE(summary.find("VRAM"), std::string::npos);
    EXPECT_NE(summary.find("RAM"), std::string::npos);
}

TEST(LeakDetectorTest, CalculateDriftEmpty) {
    LeakDetector detector;
    std::vector<size_t> values;
    int64_t drift = detector.calculate_drift(values);
    EXPECT_EQ(drift, 0);
}

TEST(LeakDetectorTest, CalculateDriftSingle) {
    LeakDetector detector;
    std::vector<size_t> values = {100};
    int64_t drift = detector.calculate_drift(values);
    EXPECT_EQ(drift, 0);
}

TEST(LeakDetectorTest, CalculateDriftPositive) {
    LeakDetector detector;
    std::vector<size_t> values = {100, 200};
    int64_t drift = detector.calculate_drift(values);
    EXPECT_EQ(drift, 100);
}

TEST(LeakDetectorTest, CalculateDriftNegative) {
    LeakDetector detector;
    std::vector<size_t> values = {200, 100};
    int64_t drift = detector.calculate_drift(values);
    EXPECT_EQ(drift, -100);
}

TEST(LeakDetectorTest, CalculateSlopeEmpty) {
    LeakDetector detector;
    std::vector<std::pair<float, size_t>> data;
    float slope = detector.calculate_slope(data);
    EXPECT_FLOAT_EQ(slope, 0.0f);
}

TEST(LeakDetectorTest, CalculateSlopeSingle) {
    LeakDetector detector;
    std::vector<std::pair<float, size_t>> data = {{0.0f, 100}};
    float slope = detector.calculate_slope(data);
    EXPECT_FLOAT_EQ(slope, 0.0f);
}

TEST(LeakDetectorTest, CalculateSlopeLinear) {
    LeakDetector detector;
    std::vector<std::pair<float, size_t>> data = {
        {0.0f, 0},
        {1.0f, 100},
        {2.0f, 200}
    };
    float slope = detector.calculate_slope(data);
    EXPECT_NEAR(slope, 100.0f, 1.0f);
}

TEST(StressRunnerTest, RampBasic) {
    MockStressServer mock;
    ASSERT_TRUE(mock.start());

    StressRunner runner;
    StressConfig cfg = test_config(mock.url());
    cfg.ramp_step_seconds = 1;
    cfg.ramp_max_users = 2;

    StressResult result = runner.run_ramp(cfg);
    mock.stop();

    EXPECT_GT(result.total_requests, 0);
    EXPECT_EQ(result.total_errors, 0);
    EXPECT_FLOAT_EQ(result.error_rate, 0.0f);
    // One timeline entry per ramp step (users 1 and 2)
    EXPECT_EQ(result.tps_timeline.size(), 2u);
    EXPECT_GT(mock.request_count(), 0);
}

TEST(StressRunnerTest, SoakNoLeakStable) {
    MockStressServer mock;
    ASSERT_TRUE(mock.start());

    StressRunner runner;
    StressConfig cfg = test_config(mock.url());
    cfg.concurrent_users = 2;
    cfg.duration_seconds = 2;
    cfg.soak_sample_interval_seconds = 1;

    StressResult result = runner.run_soak(cfg);
    mock.stop();

    EXPECT_GT(result.total_requests, 0);
    EXPECT_EQ(result.total_errors, 0);
    EXPECT_FLOAT_EQ(result.error_rate, 0.0f);
    EXPECT_GT(result.avg_tps, 0.0f);
    EXPECT_FALSE(result.tps_timeline.empty());
}

TEST(StressRunnerTest, SpikeRecovery) {
    MockStressServer mock;
    ASSERT_TRUE(mock.start());

    StressRunner runner;
    StressConfig cfg = test_config(mock.url());
    cfg.spike_baseline_users = 1;
    cfg.spike_users = 2;
    cfg.spike_phase_seconds = 1;
    cfg.spike_duration_seconds = 1;

    StressResult result = runner.run_spike(cfg);
    mock.stop();

    EXPECT_GT(result.total_requests, 0);
    // Mock server never fails: recovery phase must produce zero errors
    EXPECT_EQ(result.total_errors, 0);
    EXPECT_FLOAT_EQ(result.error_rate, 0.0f);
    EXPECT_GT(result.avg_tps, 0.0f);
}

TEST(StressRunnerTest, ChaosNoCrash) {
    MockStressServer mock;
    ASSERT_TRUE(mock.start());

    StressRunner runner;
    StressConfig cfg = test_config(mock.url());
    cfg.chaos_max_tokens = 2;
    cfg.chaos_long_prompt_chars = 1000;

    int before = mock.request_count();
    EXPECT_NO_THROW(runner.run_chaos(cfg));
    mock.stop();

    // Chaos sends 3 probes: empty messages, long context, malformed JSON
    EXPECT_EQ(mock.request_count() - before, 3);
}

}  // namespace stress
}  // namespace serius
