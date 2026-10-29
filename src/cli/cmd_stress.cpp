#include "cli/cmd_stress.h"
#include "cli/args_parser.h"
#include "stress/stress_runner.h"
#include "stress/report.h"
#include "stress/leak_detector.h"

#include <iostream>
#include <fstream>

namespace serius {
namespace cli {

int cmd_stress(int argc, char **argv)
{
    ArgsParser parser("serius stress", "Run stress tests on the server");

    parser.add_flag({"type", "t", "Test type (ramp, soak, spike, chaos)", true, "ramp"});
    parser.add_flag({"server", "s", "Server URL", true, "http://localhost:8080"});
    parser.add_flag({"users", "u", "Concurrent users", true, "8"});
    parser.add_flag({"duration", "d", "Test duration in seconds", true, "60"});
    parser.add_flag({"max-users", "", "Maximum users for ramp test", true, "128"});
    parser.add_flag({"spike-users", "", "Users for spike phase", true, "64"});
    parser.add_flag({"spike-duration", "", "Spike phase duration in seconds", true, "60"});
    parser.add_flag({"prompt", "p", "Prompt text", true, "Explain quantum computing in detail..."});
    parser.add_flag({"max-tokens", "n", "Maximum tokens to generate", true, "256"});
    parser.add_flag({"output", "o", "Output file path for report", true, "stress_report.txt"});
    parser.add_flag({"format", "f", "Report format (ascii, json)", true, "ascii"});

    if (!parser.parse(argc, argv)) {
        std::cerr << "Error: " << parser.error() << "\n";
        parser.print_help();
        return 1;
    }

    if (parser.has("help")) {
        parser.print_help();
        return 0;
    }

    std::string test_type = parser.get("type", "ramp");
    std::string server_url = parser.get("server", "http://localhost:8080");
    int users = parser.get_int("users", 8);
    int duration = parser.get_int("duration", 60);
    int max_users = parser.get_int("max-users", 128);
    int spike_users = parser.get_int("spike-users", 64);
    int spike_duration = parser.get_int("spike-duration", 60);
    std::string prompt = parser.get("prompt", "Explain quantum computing in detail...");
    int max_tokens = parser.get_int("max-tokens", 256);
    std::string output_path = parser.get("output", "stress_report.txt");
    std::string format_str = parser.get("format", "ascii");

    stress::ReportFormat format = stress::ReportFormat::ASCII;
    if (format_str == "json") {
        format = stress::ReportFormat::JSON;
    }

    stress::StressConfig config;
    config.server_url = server_url;
    config.concurrent_users = users;
    config.duration_seconds = duration;
    config.ramp_max_users = max_users;
    config.spike_users = spike_users;
    config.spike_duration_seconds = spike_duration;
    config.prompt = prompt;
    config.max_tokens = max_tokens;

    std::cout << "Serius Stress Testing\n";
    std::cout << "=====================\n";
    std::cout << "Test type: " << test_type << "\n";
    std::cout << "Server: " << server_url << "\n";
    std::cout << "Users: " << users << "\n";
    std::cout << "Duration: " << duration << " seconds\n";
    std::cout << "Max tokens: " << max_tokens << "\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "Format: " << format_str << "\n\n";

    stress::StressRunner runner;
    stress::StressResult result;

    if (test_type == "ramp") {
        std::cout << "Running ramp test...\n";
        result = runner.run_ramp(config);
    } else if (test_type == "soak") {
        std::cout << "Running soak test...\n";
        result = runner.run_soak(config);
    } else if (test_type == "spike") {
        std::cout << "Running spike test...\n";
        result = runner.run_spike(config);
    } else if (test_type == "chaos") {
        std::cout << "Running chaos test...\n";
        runner.run_chaos(server_url);
        std::cout << "Chaos test complete!\n";
        return 0;
    } else {
        std::cerr << "Unknown test type: " << test_type << "\n";
        parser.print_help();
        return 1;
    }

    std::cout << "\nGenerating report...\n";
    stress::save_report(output_path, result, config, format);

    // Run leak detection for soak tests
    if (test_type == "soak" && !result.vram_timeline.empty() && !result.ram_timeline.empty()) {
        std::cout << "Running leak detection...\n";
        std::vector<stress::MemorySample> samples;
        for (size_t i = 0; i < result.vram_timeline.size() && i < result.ram_timeline.size(); ++i) {
            samples.push_back({
                result.vram_timeline[i].first,
                result.vram_timeline[i].second,
                result.ram_timeline[i].second
            });
        }

        float duration_hours = static_cast<float>(duration) / 3600.0f;
        stress::LeakDetector detector;
        auto analysis = detector.analyze(samples, duration_hours);
        std::cout << "\n" << analysis.summary << "\n";
    }

    std::cout << "Stress test complete!\n";
    return 0;
}

}  // namespace cli
}  // namespace serius
