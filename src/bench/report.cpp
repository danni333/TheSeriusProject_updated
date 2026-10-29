#include "report.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace serius {
namespace bench {

std::string generate_report(const LatencyResult &latency,
                           const ThroughputResult &throughput,
                           const MemoryProfile &memory,
                           ReportFormat format)
{
    switch (format) {
        case ReportFormat::JSON:
        {
            nlohmann::json j;
            j["latency"] = {
                {"ttft_ms", latency.ttft_ms},
                {"decode_tps", latency.decode_tps},
                {"prefill_tps", latency.prefill_tps},
                {"prompt_tokens", latency.prompt_tokens},
                {"generated_tokens", latency.generated_tokens},
                {"total_ms", latency.total_ms}
            };
            j["throughput"] = {
                {"requests_per_sec", throughput.requests_per_sec},
                {"total_tokens_per_sec", throughput.total_tokens_per_sec},
                {"avg_latency_ms", throughput.avg_latency_ms},
                {"p99_latency_ms", throughput.p99_latency_ms},
                {"concurrent_users", throughput.concurrent_users},
                {"total_requests", throughput.total_requests},
                {"total_tokens", throughput.total_tokens}
            };
            j["memory"] = {
                {"vram_model_bytes", memory.vram_model_bytes},
                {"vram_kv_cache_bytes", memory.vram_kv_cache_bytes},
                {"vram_activations_bytes", memory.vram_activations_bytes},
                {"vram_peak_bytes", memory.vram_peak_bytes},
                {"ram_peak_bytes", memory.ram_peak_bytes}
            };
            return j.dump(2);
        }
        case ReportFormat::MARKDOWN:
        {
            std::ostringstream oss;
            oss << "# Benchmark Report\n\n";
            oss << "## Latency Metrics\n\n";
            oss << "| Metric | Value |\n";
            oss << "|--------|-------|\n";
            oss << "| Time to First Token | " << std::fixed << std::setprecision(2) << latency.ttft_ms << " ms |\n";
            oss << "| Prefill TPS | " << std::fixed << std::setprecision(2) << latency.prefill_tps << " tok/s |\n";
            oss << "| Decode TPS | " << std::fixed << std::setprecision(2) << latency.decode_tps << " tok/s |\n";
            oss << "| Prompt Tokens | " << latency.prompt_tokens << " |\n";
            oss << "| Generated Tokens | " << latency.generated_tokens << " |\n";
            oss << "| Total Time | " << std::fixed << std::setprecision(2) << latency.total_ms << " ms |\n\n";

            oss << "## Throughput Metrics\n\n";
            oss << "| Metric | Value |\n";
            oss << "|--------|-------|\n";
            oss << "| Requests/sec | " << std::fixed << std::setprecision(2) << throughput.requests_per_sec << " |\n";
            oss << "| Tokens/sec | " << std::fixed << std::setprecision(2) << throughput.total_tokens_per_sec << " |\n";
            oss << "| Avg Latency | " << std::fixed << std::setprecision(2) << throughput.avg_latency_ms << " ms |\n";
            oss << "| P99 Latency | " << std::fixed << std::setprecision(2) << throughput.p99_latency_ms << " ms |\n";
            oss << "| Concurrent Users | " << throughput.concurrent_users << " |\n";
            oss << "| Total Requests | " << throughput.total_requests << " |\n";
            oss << "| Total Tokens | " << throughput.total_tokens << " |\n\n";

            oss << "## Memory Profile\n\n";
            oss << "| Metric | Value |\n";
            oss << "|--------|-------|\n";
            oss << "| VRAM Model | " << std::fixed << std::setprecision(2)
                << (memory.vram_model_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB |\n";
            oss << "| VRAM KV Cache | " << std::fixed << std::setprecision(2)
                << (memory.vram_kv_cache_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB |\n";
            oss << "| VRAM Activations | " << std::fixed << std::setprecision(2)
                << (memory.vram_activations_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB |\n";
            oss << "| VRAM Peak | " << std::fixed << std::setprecision(2)
                << (memory.vram_peak_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB |\n";
            oss << "| RAM Peak | " << std::fixed << std::setprecision(2)
                << (memory.ram_peak_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB |\n";

            return oss.str();
        }
        case ReportFormat::CSV:
        {
            std::ostringstream oss;
            oss << "metric,value\n";
            oss << "ttft_ms," << latency.ttft_ms << "\n";
            oss << "decode_tps," << latency.decode_tps << "\n";
            oss << "prefill_tps," << latency.prefill_tps << "\n";
            oss << "prompt_tokens," << latency.prompt_tokens << "\n";
            oss << "generated_tokens," << latency.generated_tokens << "\n";
            oss << "total_ms," << latency.total_ms << "\n";
            oss << "requests_per_sec," << throughput.requests_per_sec << "\n";
            oss << "total_tokens_per_sec," << throughput.total_tokens_per_sec << "\n";
            oss << "avg_latency_ms," << throughput.avg_latency_ms << "\n";
            oss << "p99_latency_ms," << throughput.p99_latency_ms << "\n";
            oss << "concurrent_users," << throughput.concurrent_users << "\n";
            oss << "total_requests," << throughput.total_requests << "\n";
            oss << "total_tokens," << throughput.total_tokens << "\n";
            oss << "vram_model_bytes," << memory.vram_model_bytes << "\n";
            oss << "vram_kv_cache_bytes," << memory.vram_kv_cache_bytes << "\n";
            oss << "vram_activations_bytes," << memory.vram_activations_bytes << "\n";
            oss << "vram_peak_bytes," << memory.vram_peak_bytes << "\n";
            oss << "ram_peak_bytes," << memory.ram_peak_bytes << "\n";
            return oss.str();
        }
        default:
            return "Unknown format";
    }
}

void save_report(const std::string &output_path,
                 const LatencyResult &latency,
                 const ThroughputResult &throughput,
                 const MemoryProfile &memory,
                 ReportFormat format)
{
    std::string content = generate_report(latency, throughput, memory, format);
    std::ofstream file(output_path);
    if (file.is_open()) {
        file << content;
        file.close();
    } else {
        throw std::runtime_error("Could not open file: " + output_path);
    }
}

}  // namespace bench
}  // namespace serius
