#pragma once

#include <string>
#include <vector>

#include "model_interface.h"

namespace serius {
namespace agent {

/// A single sub-task produced by the planner.
struct SubTask
{
    std::string description;
    std::vector<std::string> keywords;  // Used for specialist matching.
};

/// Result of planning.
struct PlanResult
{
    std::vector<SubTask> tasks;
    std::string reasoning;  // Optional explanation.
};

/// Task decomposition planner: breaks a complex query into sub-tasks.
class Planner
{
   public:
    explicit Planner(Model &model);

    /// Decompose a user query into sub-tasks.
    PlanResult decompose(const std::string &query) const;

    /// Decompose with a system prompt override.
    PlanResult decompose(const std::string &query, const std::string &system_prompt) const;

   private:
    Model &model_;
};

}  // namespace agent
}  // namespace serius