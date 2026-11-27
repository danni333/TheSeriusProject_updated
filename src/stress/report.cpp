#include "report.h"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

namespace serius {
namespace stress {

std::string generate_report(const StressResult &result, const StressConfig &config, ReportFormat format)
{
    if (format == ReportFormat::JSON) {
        nlohmann::json j;
        j["config"] = {{"concurrent_users", config.concurrent_users},
                       {"duration_seconds", config.duration_seconds},
                       {"prompt", config.prompt},
                       {"max_tokens", config.max_tokens},
                       {"model_name", config.model_name},
                       {"server_url", config.server_url}};
        j["result"] = {{"avg_tps", result.avg_tps},
                       {"p50_latency_ms", result.p50_latency_ms},
                       {"p99_latency_ms", result.p99_latency_ms},
                       {"error_rate", result.error_rate},
                       {"total_requests", result.total_requests},
                       {"total_errors", result.total_errors},
                       {"total_tokens", result.total_tokens}};

        nlohmann::json tps_data;
        for (const auto &[time, tps] : result.tps_timeline) {
            tps_data.push_back({time, tps});
        }
        j["result"]["tps_timeline"] = tps_data;

        nlohmann::json vram_data;
        for (const auto &[time, vram] : result.vram_timeline) {
            vram_data.push_back({time, vram});
        }
        j["result"]["vram_timeline"] = vram_data;

        nlohmann::json ram_data;
        for (const auto &[time, ram] : result.ram_timeline) {
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
#if defined(_MSC_VER)
        struct tm tm_buf;
        localtime_s(&tm_buf, &now);
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);
#else
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
#endif

        ss << "═══════════════════════════════════════════════════════\n";
        ss << "  STRESS TEST REPORT — " << time_str << "\n";
        ss << "═══════════════════════════════════════════════════════\n\n";

        ss << "  Server URL:       " << config.server_url << "\n";
        ss << "  Concurrent Users: " << config.concurrent_users << "\n";
        ss << "  Duration:         " << config.duration_seconds << " seconds\n";
        ss << "  Max Tokens:       " << config.max_tokens << "\n\n";

        auto row = [&](const char *metric, const std::string &value) {
            ss << "  │ " << std::left << std::setw(12) << metric << " │ " << std::right << std::setw(10) << value
               << " │\n";
        };
        ss << "  ┌──────────────┬────────────┐\n";
        ss << "  │ Metric       │ Value      │\n";
        ss << "  ├──────────────┼────────────┤\n";
        row("Avg TPS", std::to_string(result.avg_tps));
        row("P50 Latency", std::to_string(result.p50_latency_ms) + " ms");
        row("P99 Latency", std::to_string(result.p99_latency_ms) + " ms");
        row("Error Rate", std::to_string(result.error_rate * 100.0f) + "%");
        row("Total Reqs", std::to_string(result.total_requests));
        row("Total Tokens", std::to_string(result.total_tokens));
        row("Total Errors", std::to_string(result.total_errors));
        ss << "  └──────────────┴────────────┘\n\n";

        // Memory leak detection: compare the median of the last samples
        // against the median of the whole run. Endpoint-vs-endpoint drift is
        // misleading — the first sample catches transient startup allocations
        // (model mmap/page cache) that are legitimately freed later, which
        // produced false "LEAK DETECTED" reports on negative drift.
        auto leak_check = [](const std::vector<std::pair<float, size_t>> &tl, const char *name,
                             std::ostringstream &ss) {
            if (tl.empty()) {
                return;
            }
            std::vector<size_t> vals;
            vals.reserve(tl.size());
            for (const auto &p : tl) vals.push_back(p.second);
            std::sort(vals.begin(), vals.end());
            const size_t median_all = vals[vals.size() / 2];
            // Tail = last samples in TIME order (tl is appended chronologically
            // by the runner), not the largest values — a mid-run spike that
            // later subsides must not look like a leak.
            const size_t tail_n = std::max<size_t>(1, tl.size() / 10);
            std::vector<size_t> tail;
            tail.reserve(tail_n);
            for (size_t i = tl.size() - tail_n; i < tl.size(); ++i) {
                tail.push_back(tl[i].second);
            }
            std::sort(tail.begin(), tail.end());
            const size_t median_tail = tail[tail.size() / 2];
            const int64_t drift = static_cast<int64_t>(median_tail) - static_cast<int64_t>(median_all);
            ss << "  " << name << " Analysis:\n";
            ss << "    Median: " << (median_all / 1024 / 1024) << " MB\n";
            ss << "    Tail:   " << (median_tail / 1024 / 1024) << " MB\n";
            ss << "    Drift:  " << (drift / 1024 / 1024) << " MB ";
            if (drift > 256LL * 1024 * 1024) {
                ss << "⚠️ (potential leak)\n";
            } else {
                ss << "✅ (no leak detected)\n";
            }
            ss << "\n";
        };
        {
            std::ostringstream vram_ss;
            leak_check(result.vram_timeline, "VRAM", vram_ss);
            ss << vram_ss.str();
        }
        {
            std::ostringstream ram_ss;
            leak_check(result.ram_timeline, "RAM", ram_ss);
            ss << ram_ss.str();
        }

        ss << "═══════════════════════════════════════════════════════\n";

        return ss.str();
    }
}

void save_report(const std::string &output_path, const StressResult &result, const StressConfig &config,
                 ReportFormat format)
{
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
