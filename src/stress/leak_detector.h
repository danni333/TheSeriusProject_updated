#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace serius {
namespace stress {

struct MemorySample {
    float timestamp;
    size_t vram_bytes;
    size_t ram_bytes;
};

struct LeakAnalysis {
    bool vram_leak_detected = false;
    bool ram_leak_detected = false;
    int64_t vram_drift_bytes = 0;
    int64_t ram_drift_bytes = 0;
    float vram_drift_mb_per_hour = 0.0f;
    float ram_drift_mb_per_hour = 0.0f;
    std::string summary;
};

/// Detect memory leaks from timeline samples.
class LeakDetector {
public:
    LeakDetector() = default;
    ~LeakDetector() = default;

    /// Analyze memory samples for leaks
    LeakAnalysis analyze(const std::vector<MemorySample>& samples,
                         float duration_hours,
                         size_t leak_threshold_bytes = 10 * 1024 * 1024);  // 10 MB default

    /// Generate human-readable summary
    std::string format_summary(const LeakAnalysis& analysis);

private:
    /// Calculate linear regression slope
    float calculate_slope(const std::vector<std::pair<float, size_t>>& data);

    /// Calculate drift (end - start)
    int64_t calculate_drift(const std::vector<size_t>& values);
};

}  // namespace stress
}  // namespace serius
