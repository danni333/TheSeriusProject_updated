#include "report.h"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <nlohmann/json.hpp>

namespace serius {
namespace stress {

std::string generate_report(const StressResult& result,
                           const StressConfig& config,
                           ReportFormat format) {
    if (format == ReportFormat::JSON) {
        nlohmann::json j;
        j["config"] = {
            {"concurrent_users", config.concurrent_users},
            {"duration_seconds", config.duration_seconds},
            {"prompt", config.prompt},
            {"max_tokens", config.max_tokens},
            {"server_url", config.server_url}
        };
        j["result"] = {
            {"avg_tps", result.avg_tps},
            {"p50_latency_ms", result.p50_latency_ms},
            {"p99_latency_ms", result.p99_latency_ms},
            {"error_rate", result.error_rate},
            {"total_requests", result.total_requests},
            {"total_errors", result.total_errors}
        };

        nlohmann::json tps_data;
        for (const auto& [time, tps] : result.tps_timeline) {
            tps_data.push_back({time, tps});
        }
        j["result"]["tps_timeline"] = tps_data;

        nlohmann::json vram_data;
        for (const auto& [time, vram] : result.vram_timeline) {
            vram_data.push_back({time, vram});
        }
        j["result"]["vram_timeline"] = vram_data;

        nlohmann::json ram_data;
        for (const auto& [time, ram] : result.ram_timeline) {
            ram_data.push_back({time, ram});
        }
        j["result"]["ram_timeline"] = ram_data;

        return j.dump(2);
    } else {
        // ASCII format
        std::stringstream ss;

        // Get current timestamp
        auto now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        ss << "═══════════════════════════════════════════════════════\n";
        ss << "  STRESS TEST REPORT — " << time_str << "\n";
        ss << "═══════════════════════════════════════════════════════\n\n";

        ss << "  Server URL:       " << config.server_url << "\n";
        ss << "  Concurrent Users: " << config.concurrent_users << "\n";
        ss << "  Duration:         " << config.duration_seconds << " seconds\n";
        ss << "  Max Tokens:       " << config.max_tokens << "\n\n";

        ss << "  ┌──────────────┬────────────┬────────────┬────────────┬──────────┐\n";
        ss << "  │ Metric       │ Value      │            │            │          │\n";
        ss << "  ├──────────────┼────────────┼────────────┼────────────┼──────────┤\n";
        ss << "  │ Avg TPS      │ " << std::setw(10) << std::fixed << std::setprecision(1) << result.avg_tps << " │            │            │          │\n";
        ss << "  │ P50 Latency  │ " << std::setw(10) << std::fixed << std::setprecision(1) << result.p50_latency_ms << " ms      │            │            │          │\n";
        ss << "  │ P99 Latency  │ " << std::setw(10) << std::fixed << std::setprecision(1) << result.p99_latency_ms << " ms      │            │            │          │\n";
        ss << "  │ Error Rate   │ " << std::setw(10) << std::fixed << std::setprecision(2) << (result.error_rate * 100) << "% │            │            │          │\n";
        ss << "  │ Total Reqs   │ " << std::setw(10) << result.total_requests << " │            │            │          │\n";
        ss << "  │ Total Errors │ " << std::setw(10) << result.total_errors << " │            │            │          │\n";
        ss << "  └──────────────┴────────────┴────────────┴────────────┴──────────┘\n\n";

        // Memory leak detection
        if (!result.vram_timeline.empty()) {
            size_t vram_start = result.vram_timeline.front().second;
            size_t vram_end = result.vram_timeline.back().second;
            int64_t vram_drift = static_cast<int64_t>(vram_end) - static_cast<int64_t>(vram_start);

            ss << "  VRAM Analysis:\n";
            ss << "    Start: " << (vram_start / 1024 / 1024) << " MB\n";
            ss << "    End:   " << (vram_end / 1024 / 1024) << " MB\n";
            ss << "    Drift: " << (vram_drift / 1024 / 1024) << " MB ";
            if (std::abs(vram_drift) < 1024 * 1024) {
                ss << "✅ (no leak detected)\n";
            } else {
                ss << "⚠️ (potential leak)\n";
            }
            ss << "\n";
        }

        if (!result.ram_timeline.empty()) {
            size_t ram_start = result.ram_timeline.front().second;
            size_t ram_end = result.ram_timeline.back().second;
            int64_t ram_drift = static_cast<int64_t>(ram_end) - static_cast<int64_t>(ram_start);

            ss << "  RAM Analysis:\n";
            ss << "    Start: " << (ram_start / 1024 / 1024) << " MB\n";
            ss << "    End:   " << (ram_end / 1024 / 1024) << " MB\n";
            ss << "    Drift: " << (ram_drift / 1024 / 1024) << " MB ";
            if (std::abs(ram_drift) < 1024 * 1024) {
                ss << "✅ (no leak detected)\n";
            } else {
                ss << "⚠️ (potential leak)\n";
            }
            ss << "\n";
        }

        ss << "═══════════════════════════════════════════════════════\n";

        return ss.str();
    }
}

void save_report(const std::string& output_path,
                 const StressResult& result,
                 const StressConfig& config,
                 ReportFormat format) {
    std::string report = generate_report(result, config, format);

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    out << report;
    out.close();
}

}  // namespace stress
}  // namespace serius
