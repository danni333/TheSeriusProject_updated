#include <gtest/gtest.h>
#include "stress/stress_runner.h"
#include "stress/report.h"
#include "stress/leak_detector.h"

namespace serius {
namespace stress {

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

}  // namespace stress
}  // namespace serius
