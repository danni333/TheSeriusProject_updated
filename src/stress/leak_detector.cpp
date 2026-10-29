#include "leak_detector.h"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <numeric>

namespace serius {
namespace stress {

LeakAnalysis LeakDetector::analyze(const std::vector<MemorySample>& samples,
                                   float duration_hours,
                                   size_t leak_threshold_bytes) {
    LeakAnalysis analysis;

    if (samples.size() < 2) {
        analysis.summary = "Insufficient data for leak detection";
        return analysis;
    }

    // Extract VRAM and RAM timelines
    std::vector<std::pair<float, size_t>> vram_data;
    std::vector<std::pair<float, size_t>> ram_data;
    std::vector<size_t> vram_values;
    std::vector<size_t> ram_values;

    for (const auto& sample : samples) {
        vram_data.push_back({sample.timestamp, sample.vram_bytes});
        ram_data.push_back({sample.timestamp, sample.ram_bytes});
        vram_values.push_back(sample.vram_bytes);
        ram_values.push_back(sample.ram_bytes);
    }

    // Calculate drifts
    analysis.vram_drift_bytes = calculate_drift(vram_values);
    analysis.ram_drift_bytes = calculate_drift(ram_values);

    // Check for leaks
    analysis.vram_leak_detected = std::abs(analysis.vram_drift_bytes) > static_cast<int64_t>(leak_threshold_bytes);
    analysis.ram_leak_detected = std::abs(analysis.ram_drift_bytes) > static_cast<int64_t>(leak_threshold_bytes);

    // Calculate drift rate per hour
    if (duration_hours > 0) {
        analysis.vram_drift_mb_per_hour = static_cast<float>(analysis.vram_drift_bytes) / (1024.0f * 1024.0f) / duration_hours;
        analysis.ram_drift_mb_per_hour = static_cast<float>(analysis.ram_drift_bytes) / (1024.0f * 1024.0f) / duration_hours;
    }

    analysis.summary = format_summary(analysis);

    return analysis;
}

std::string LeakDetector::format_summary(const LeakAnalysis& analysis) {
    std::stringstream ss;

    ss << "Memory Leak Analysis:\n";
    ss << "=====================\n\n";

    ss << "VRAM:\n";
    ss << "  Drift: " << std::fixed << std::setprecision(2)
       << (analysis.vram_drift_bytes / 1024.0 / 1024.0) << " MB\n";
    ss << "  Rate:  " << std::fixed << std::setprecision(2)
       << analysis.vram_drift_mb_per_hour << " MB/hour\n";
    if (analysis.vram_leak_detected) {
        ss << "  Status: ⚠️ LEAK DETECTED\n";
    } else {
        ss << "  Status: ✅ No leak detected\n";
    }
    ss << "\n";

    ss << "RAM:\n";
    ss << "  Drift: " << std::fixed << std::setprecision(2)
       << (analysis.ram_drift_bytes / 1024.0 / 1024.0) << " MB\n";
    ss << "  Rate:  " << std::fixed << std::setprecision(2)
       << analysis.ram_drift_mb_per_hour << " MB/hour\n";
    if (analysis.ram_leak_detected) {
        ss << "  Status: ⚠️ LEAK DETECTED\n";
    } else {
        ss << "  Status: ✅ No leak detected\n";
    }

    return ss.str();
}

float LeakDetector::calculate_slope(const std::vector<std::pair<float, size_t>>& data) {
    if (data.size() < 2) return 0.0f;

    // Simple linear regression: y = mx + b
    float n = static_cast<float>(data.size());
    float sum_x = 0.0f, sum_y = 0.0f;
    float sum_xy = 0.0f, sum_x2 = 0.0f;

    for (const auto& [x, y] : data) {
        sum_x += x;
        sum_y += static_cast<float>(y);
        sum_xy += x * static_cast<float>(y);
        sum_x2 += x * x;
    }

    float denominator = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < 1e-6f) return 0.0f;

    float slope = (n * sum_xy - sum_x * sum_y) / denominator;
    return slope;
}

int64_t LeakDetector::calculate_drift(const std::vector<size_t>& values) {
    if (values.size() < 2) return 0;

    size_t start = values.front();
    size_t end = values.back();

    return static_cast<int64_t>(end) - static_cast<int64_t>(start);
}

}  // namespace stress
}  // namespace serius
