#pragma once

#include <string>

#include "latency.h"
#include "memory.h"
#include "throughput.h"

namespace serius {
namespace bench {

enum class ReportFormat { JSON, MARKDOWN, CSV };

/// Generate benchmark report in specified format.
/// Saves results to file or returns as string.
std::string generate_report(const LatencyResult &latency, const ThroughputResult &throughput,
                            const MemoryProfile &memory, ReportFormat format = ReportFormat::MARKDOWN);

/// Save benchmark report to file.
void save_report(const std::string &output_path, const LatencyResult &latency, const ThroughputResult &throughput,
                 const MemoryProfile &memory, ReportFormat format = ReportFormat::MARKDOWN);

}  // namespace bench
}  // namespace serius
