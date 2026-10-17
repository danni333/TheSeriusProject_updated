#pragma once

#include <string>
#include <vector>

#include "eval_suite.h"

namespace serius {
namespace eval {

void save_json_report(const std::vector<EvalResult> &results, const std::string &path);
void save_csv_report(const std::vector<EvalResult> &results, const std::string &path);

}  // namespace eval
}  // namespace serius
