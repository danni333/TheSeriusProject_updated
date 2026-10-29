#pragma once

#include <string>
#include <vector>

#include "model_interface.h"
#include "specialist.h"

namespace serius {
namespace agent {

/// Task to be executed by a specialist.
struct Task
{
    std::string id;
    std::string agent_name;
    std::string prompt;
};

/// Result of a task execution.
struct TaskResult
{
    std::string task_id;
    std::string agent_name;
    std::string output;
    bool success;
};

/// Supervisor pattern: delegates tasks to specialists and synthesizes results.
class Supervisor
{
   public:
    Supervisor(Model &core_model);

    /// Register a specialist agent.
    void register_specialist(Specialist *agent);

    /// Process a user query: plan, delegate, synthesize.
    std::string process(const std::string &query);

    const std::vector<Specialist *> &specialists() const { return specialists_; }

   private:
    Model &core_model_;
    std::vector<Specialist *> specialists_;

    /// Generate a plan from the core model.
    std::vector<Task> plan_tasks(const std::string &query);

    /// Execute a list of tasks, returning results.
    std::vector<TaskResult> execute_tasks(const std::vector<Task> &tasks);

    /// Synthesize results into a final answer.
    std::string synthesize(const std::string &query, const std::vector<TaskResult> &results);
};

}  // namespace agent
}  // namespace serius