#pragma once

#include <string>

namespace serius {
namespace reasoning {

/// Adaptive compute budget based on query complexity.
class ComputeBudget
{
   public:
    /// Estimate the thinking budget (in tokens) for a query.
    static int estimate_budget(const std::string &query);

    /// Default budgets.
    static constexpr int kSimple = 512;
    static constexpr int kMedium = 1024;
    static constexpr int kComplex = 4096;
};

}  // namespace reasoning
}  // namespace serius
