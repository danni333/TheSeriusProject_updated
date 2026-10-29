#include <gtest/gtest.h>
#include "bench/latency.h"
#include "bench/throughput.h"
#include "bench/memory.h"
#include "bench/report.h"

namespace serius {
namespace bench {

TEST(BenchLatency, BenchSingleBasic)
{
    LatencyResult result = bench_single("dummy_model.gguf", "Hello world", 256);

    EXPECT_GE(result.prompt_tokens, 0);
    EXPECT_GE(result.generated_tokens, 0);
    EXPECT_GE(result.ttft_ms, 0.0f);
    EXPECT_GE(result.decode_tps, 0.0f);
    EXPECT_GE(result.prefill_tps, 0.0f);
    EXPECT_GE(result.total_ms, 0.0f);
}

TEST(BenchLatency, LatencyResultFields)
{
    LatencyResult result;
    result.ttft_ms = 150.0f;
    result.decode_tps = 80.0f;
    result.prefill_tps = 2000.0f;
    result.prompt_tokens = 100;
    result.generated_tokens = 256;
    result.total_ms = 1500.0f;

    EXPECT_FLOAT_EQ(result.ttft_ms, 150.0f);
    EXPECT_FLOAT_EQ(result.decode_tps, 80.0f);
    EXPECT_FLOAT_EQ(result.prefill_tps, 2000.0f);
    EXPECT_EQ(result.prompt_tokens, 100);
    EXPECT_EQ(result.generated_tokens, 256);
    EXPECT_FLOAT_EQ(result.total_ms, 1500.0f);
}

TEST(BenchThroughput, BenchThroughputBasic)
{
    std::vector<std::string> prompts = {
        "Hello world",
        "What is AI?",
        "Explain quantum computing"
    };

    ThroughputResult result = bench_throughput("dummy_model.gguf", prompts, 2, 3, 256);

    EXPECT_GE(result.concurrent_users, 0);
    EXPECT_GE(result.total_requests, 0);
    EXPECT_GE(result.total_tokens, 0);
    EXPECT_GE(result.requests_per_sec, 0.0f);
    EXPECT_GE(result.total_tokens_per_sec, 0.0f);
    EXPECT_GE(result.avg_latency_ms, 0.0f);
    EXPECT_GE(result.p99_latency_ms, 0.0f);
}

TEST(BenchThroughput, ThroughputResultFields)
{
    ThroughputResult result;
    result.requests_per_sec = 10.5f;
    result.total_tokens_per_sec = 500.0f;
    result.avg_latency_ms = 200.0f;
    result.p99_latency_ms = 300.0f;
    result.concurrent_users = 4;
    result.total_requests = 40;
    result.total_tokens = 1000;

    EXPECT_FLOAT_EQ(result.requests_per_sec, 10.5f);
    EXPECT_FLOAT_EQ(result.total_tokens_per_sec, 500.0f);
    EXPECT_FLOAT_EQ(result.avg_latency_ms, 200.0f);
    EXPECT_FLOAT_EQ(result.p99_latency_ms, 300.0f);
    EXPECT_EQ(result.concurrent_users, 4);
    EXPECT_EQ(result.total_requests, 40);
    EXPECT_EQ(result.total_tokens, 1000);
}

TEST(BenchMemory, ProfileMemoryBasic)
{
    MemoryProfile profile = profile_memory("dummy_model.gguf", 512, 1);

    EXPECT_GE(profile.vram_model_bytes, 0ull);
    EXPECT_GE(profile.vram_kv_cache_bytes, 0ull);
    EXPECT_GE(profile.vram_activations_bytes, 0ull);
    EXPECT_GE(profile.vram_peak_bytes, 0ull);
    EXPECT_GE(profile.ram_peak_bytes, 0ull);
}

TEST(BenchMemory, MemoryProfileFields)
{
    MemoryProfile profile;
    profile.vram_model_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    profile.vram_kv_cache_bytes = 512ull * 1024ull * 1024ull;
    profile.vram_activations_bytes = 256ull * 1024ull * 1024ull;
    profile.vram_peak_bytes = 5ull * 1024ull * 1024ull * 1024ull;
    profile.ram_peak_bytes = 2ull * 1024ull * 1024ull * 1024ull;

    EXPECT_EQ(profile.vram_model_bytes, 4ull * 1024ull * 1024ull * 1024ull);
    EXPECT_EQ(profile.vram_kv_cache_bytes, 512ull * 1024ull * 1024ull);
    EXPECT_EQ(profile.vram_activations_bytes, 256ull * 1024ull * 1024ull);
    EXPECT_EQ(profile.vram_peak_bytes, 5ull * 1024ull * 1024ull * 1024ull);
    EXPECT_EQ(profile.ram_peak_bytes, 2ull * 1024ull * 1024ull * 1024ull);
}

TEST(BenchReport, GenerateReportJSON)
{
    LatencyResult latency;
    latency.ttft_ms = 150.0f;
    latency.decode_tps = 80.0f;
    latency.prefill_tps = 2000.0f;
    latency.prompt_tokens = 100;
    latency.generated_tokens = 256;
    latency.total_ms = 1500.0f;

    ThroughputResult throughput;
    throughput.requests_per_sec = 10.0f;
    throughput.total_tokens_per_sec = 500.0f;
    throughput.avg_latency_ms = 200.0f;
    throughput.p99_latency_ms = 300.0f;
    throughput.concurrent_users = 4;
    throughput.total_requests = 40;
    throughput.total_tokens = 1000;

    MemoryProfile memory;
    memory.vram_model_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    memory.vram_kv_cache_bytes = 512ull * 1024ull * 1024ull;
    memory.vram_activations_bytes = 256ull * 1024ull * 1024ull;
    memory.vram_peak_bytes = 5ull * 1024ull * 1024ull * 1024ull;
    memory.ram_peak_bytes = 2ull * 1024ull * 1024ull * 1024ull;

    std::string json_report = generate_report(latency, throughput, memory, ReportFormat::JSON);

    EXPECT_FALSE(json_report.empty());
    EXPECT_NE(json_report.find("\"latency\""), std::string::npos);
    EXPECT_NE(json_report.find("\"throughput\""), std::string::npos);
    EXPECT_NE(json_report.find("\"memory\""), std::string::npos);
}

TEST(BenchReport, GenerateReportMarkdown)
{
    LatencyResult latency;
    latency.ttft_ms = 150.0f;
    latency.decode_tps = 80.0f;
    latency.prefill_tps = 2000.0f;
    latency.prompt_tokens = 100;
    latency.generated_tokens = 256;
    latency.total_ms = 1500.0f;

    ThroughputResult throughput;
    throughput.requests_per_sec = 10.0f;
    throughput.total_tokens_per_sec = 500.0f;
    throughput.avg_latency_ms = 200.0f;
    throughput.p99_latency_ms = 300.0f;
    throughput.concurrent_users = 4;
    throughput.total_requests = 40;
    throughput.total_tokens = 1000;

    MemoryProfile memory;
    memory.vram_model_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    memory.vram_kv_cache_bytes = 512ull * 1024ull * 1024ull;
    memory.vram_activations_bytes = 256ull * 1024ull * 1024ull;
    memory.vram_peak_bytes = 5ull * 1024ull * 1024ull * 1024ull;
    memory.ram_peak_bytes = 2ull * 1024ull * 1024ull * 1024ull;

    std::string md_report = generate_report(latency, throughput, memory, ReportFormat::MARKDOWN);

    EXPECT_FALSE(md_report.empty());
    EXPECT_NE(md_report.find("# Benchmark Report"), std::string::npos);
    EXPECT_NE(md_report.find("## Latency Metrics"), std::string::npos);
    EXPECT_NE(md_report.find("## Throughput Metrics"), std::string::npos);
    EXPECT_NE(md_report.find("## Memory Profile"), std::string::npos);
}

TEST(BenchReport, GenerateReportCSV)
{
    LatencyResult latency;
    latency.ttft_ms = 150.0f;
    latency.decode_tps = 80.0f;
    latency.prefill_tps = 2000.0f;
    latency.prompt_tokens = 100;
    latency.generated_tokens = 256;
    latency.total_ms = 1500.0f;

    ThroughputResult throughput;
    throughput.requests_per_sec = 10.0f;
    throughput.total_tokens_per_sec = 500.0f;
    throughput.avg_latency_ms = 200.0f;
    throughput.p99_latency_ms = 300.0f;
    throughput.concurrent_users = 4;
    throughput.total_requests = 40;
    throughput.total_tokens = 1000;

    MemoryProfile memory;
    memory.vram_model_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    memory.vram_kv_cache_bytes = 512ull * 1024ull * 1024ull;
    memory.vram_activations_bytes = 256ull * 1024ull * 1024ull;
    memory.vram_peak_bytes = 5ull * 1024ull * 1024ull * 1024ull;
    memory.ram_peak_bytes = 2ull * 1024ull * 1024ull * 1024ull;

    std::string csv_report = generate_report(latency, throughput, memory, ReportFormat::CSV);

    EXPECT_FALSE(csv_report.empty());
    EXPECT_NE(csv_report.find("metric,value"), std::string::npos);
    EXPECT_NE(csv_report.find("ttft_ms,"), std::string::npos);
    EXPECT_NE(csv_report.find("decode_tps,"), std::string::npos);
}

}  // namespace bench
}  // namespace serius
