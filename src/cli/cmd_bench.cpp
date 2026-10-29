#include "cli/cmd_bench.h"
#include "cli/args_parser.h"
#include "bench/latency.h"
#include "bench/throughput.h"
#include "bench/memory.h"
#include "bench/report.h"

#include <iostream>
#include <vector>

namespace serius {
namespace cli {

int cmd_bench(int argc, char **argv)
{
    ArgsParser parser("serius bench", "Run performance benchmarks on the model");

    parser.add_flag({"model", "m", "Path to model file or directory", true, "./models/model.gguf"});
    parser.add_flag({"prompt", "p", "Prompt text for latency benchmark", true, "The quick brown fox jumps over the lazy dog."});
    parser.add_flag({"max-tokens", "n", "Maximum tokens to generate", true, "256"});
    parser.add_flag({"concurrent", "c", "Number of concurrent users for throughput", true, "4"});
    parser.add_flag({"requests", "r", "Requests per user for throughput", true, "10"});
    parser.add_flag({"output", "o", "Output file path for report", true, "benchmark_report.md"});
    parser.add_flag({"format", "f", "Report format (json, markdown, csv)", true, "markdown"});
    parser.add_flag({"latency-only", "", "Run only latency benchmark", false, ""});
    parser.add_flag({"throughput-only", "", "Run only throughput benchmark", false, ""});
    parser.add_flag({"memory-only", "", "Run only memory profiling", false, ""});

    if (!parser.parse(argc, argv)) {
        std::cerr << "Error: " << parser.error() << "\n";
        parser.print_help();
        return 1;
    }

    if (parser.has("help")) {
        parser.print_help();
        return 0;
    }

    std::string model_path = parser.get("model");
    std::string prompt = parser.get("prompt");
    int max_tokens = parser.get_int("max-tokens", 256);
    int concurrent = parser.get_int("concurrent", 4);
    int requests = parser.get_int("requests", 10);
    std::string output_path = parser.get("output", "benchmark_report.md");
    std::string format_str = parser.get("format", "markdown");

    bool latency_only = parser.has("latency-only");
    bool throughput_only = parser.has("throughput-only");
    bool memory_only = parser.has("memory-only");

    bench::ReportFormat format = bench::ReportFormat::MARKDOWN;
    if (format_str == "json") {
        format = bench::ReportFormat::JSON;
    } else if (format_str == "csv") {
        format = bench::ReportFormat::CSV;
    }

    std::cout << "Serius Performance Benchmark\n";
    std::cout << "============================\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Prompt: \"" << prompt << "\"\n";
    std::cout << "Max tokens: " << max_tokens << "\n";
    std::cout << "Concurrent users: " << concurrent << "\n";
    std::cout << "Requests per user: " << requests << "\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "Format: " << format_str << "\n\n";

    bench::LatencyResult latency;
    bench::ThroughputResult throughput;
    bench::MemoryProfile memory;

    // Run latency benchmark
    if (!throughput_only && !memory_only) {
        std::cout << "Running latency benchmark...\n";
        latency = bench::bench_single(model_path, prompt, max_tokens);
        std::cout << "\n";
    }

    // Run throughput benchmark
    if (!latency_only && !memory_only) {
        std::cout << "Running throughput benchmark...\n";
        std::vector<std::string> prompts = {prompt, "Hello world!", "What is AI?", "Explain quantum computing."};
        throughput = bench::bench_throughput(model_path, prompts, concurrent, requests, max_tokens);
        std::cout << "\n";
    }

    // Run memory profiling
    if (!latency_only && !throughput_only) {
        std::cout << "Running memory profiling...\n";
        memory = bench::profile_memory(model_path, 512, 1);
        std::cout << "\n";
    }

    // Generate and save report
    std::cout << "Generating report...\n";
    bench::save_report(output_path, latency, throughput, memory, format);

    std::cout << "Benchmark complete!\n";
    return 0;
}

}  // namespace cli
}  // namespace serius
