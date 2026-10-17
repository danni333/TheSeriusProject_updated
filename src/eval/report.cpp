#include "report.h"

#include <fstream>

namespace serius {
namespace eval {

void save_json_report(const std::vector<EvalResult> &results, const std::string &path)
{
    std::ofstream f(path);
    if (!f) {
        return;
    }
    f << "[\n";
    for (size_t i = 0; i < results.size(); ++i) {
        f << "  {\"benchmark\": \"" << results[i].benchmark
          << "\", \"correct\": " << results[i].correct
          << ", \"total\": " << results[i].total
          << ", \"accuracy\": " << results[i].accuracy() << "}";
        if (i + 1 < results.size()) {
            f << ",";
        }
        f << "\n";
    }
    f << "]\n";
}

void save_csv_report(const std::vector<EvalResult> &results, const std::string &path)
{
    std::ofstream f(path);
    if (!f) {
        return;
    }
    f << "benchmark,correct,total,accuracy\n";
    for (const auto &r : results) {
        f << r.benchmark << "," << r.correct << "," << r.total << "," << r.accuracy() << "\n";
    }
}

}  // namespace eval
}  // namespace serius
