#pragma once

#include "stress_runner.h"
#include <string>

namespace serius {
namespace stress {

enum class ReportFormat {
    ASCII,
    JSON
};

/// Generate stress test report in specified format.
std::string generate_report(const StressResult& result,
                           const StressConfig& config,
                           ReportFormat format = ReportFormat::ASCII);

/// Save stress test report to file.
void save_report(const std::string& output_path,
                 const StressResult& result,
                 const StressConfig& config,
                 ReportFormat format = ReportFormat::ASCII);

}  // namespace stress
}  // namespace serius
