#include "budget.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace serius {
namespace reasoning {

namespace {

std::string to_lower(std::string s)
{
    for (auto &c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

}  // namespace

int ComputeBudget::estimate_budget(const std::string &query)
{
    std::string low = to_lower(query);

    const std::vector<std::string> math_kws = {"+",  "-", "*", "/", "=",  "calculate",
                                               "solve", "equation", "math", "percent"};
    const std::vector<std::string> code_kws = {"python", "javascript", "function",
                                               "code",   "program",  "script", "bug"};

    const int simple_budget = 512;
    const int complex_budget = 4096;

    auto contains_any = [&](const std::vector<std::string> &kws) {
        for (const auto &kw : kws) {
            if (low.find(kw) != std::string::npos) {
                return true;
            }
        }
        return false;
    };

    if (contains_any(math_kws)) {
        return complex_budget;
    }
    if (contains_any(code_kws)) {
        return complex_budget;
    }
    return simple_budget;
}

}  // namespace reasoning
}  // namespace serius
