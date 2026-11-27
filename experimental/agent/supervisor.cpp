#include "supervisor.h"
#include "specialist.h"

#include <nlohmann/json.hpp>
#include <sstream>

namespace serius {
namespace agent {

Supervisor::Supervisor(Model &core_model) : core_model_(core_model)
{
}

void Supervisor::register_specialist(Specialist *agent)
{
    specialists_.push_back(agent);
}

std::vector<Task> Supervisor::plan_tasks(const std::string &query)
{
    std::vector<Task> tasks;
    if (specialists_.empty()) {
        return tasks;
    }

    // Build a prompt listing available specialists.
    std::string prompt = "You are a supervisor. Analyze this request and decide which specialists to use.\n"
                         "Request: " +
                         query +
                         "\n\nAvailable specialists:\n";

    for (const auto *s : specialists_) {
        prompt += "- " + s->name() + ": " + s->specialty() + "\n";
    }

    prompt += "\nRespond with a JSON array of tasks. Each task: {\"agent\":\"<name>\", \"prompt\":\"<task description>\"}.\n"
              "If no specialist is needed, respond with an empty array [].";

    SamplingParams params;
    params.temperature = 0.3f;
    params.max_tokens = 512;
    std::string plan = core_model_.generate(prompt, params);

    // Parse the plan as JSON: hand-rolled quote scanning broke on escaped
    // quotes, ':' or ';' inside values, and nested objects.
    nlohmann::json plan_json;
    try {
        plan_json = nlohmann::json::parse(plan);
    }
    catch (...) {
        return tasks;  // Unparseable plan -> no tasks
    }
    if (!plan_json.is_array()) {
        return tasks;
    }

    int task_id = 0;
    for (const auto &item : plan_json) {
        if (!item.is_object()) {
            continue;
        }
        Task t;
        t.id = std::to_string(task_id++);
        t.agent_name = item.value("agent", "");
        t.prompt = item.value("prompt", "");
        if (!t.agent_name.empty()) {
            tasks.push_back(std::move(t));
        }
    }

    return tasks;
}

std::vector<TaskResult> Supervisor::execute_tasks(const std::vector<Task> &tasks)
{
    std::vector<TaskResult> results;

    for (const auto &task : tasks) {
        TaskResult result;
        result.task_id = task.id;
        result.agent_name = task.agent_name;
        result.success = false;

        for (const auto *s : specialists_) {
            if (s->name() == task.agent_name) {
                result.output = s->execute(task.prompt);
                result.success = true;
                break;
            }
        }

        if (!result.success) {
            result.output = "Specialist " + task.agent_name + " not found.";
        }

        results.push_back(std::move(result));
    }

    return results;
}

std::string Supervisor::synthesize(const std::string &query, const std::vector<TaskResult> &results)
{
    std::string prompt = "You are a supervisor. Synthesize the following results into a coherent final answer.\n\n"
                         "Original request: " +
                         query + "\n\nResults:\n";

    for (const auto &r : results) {
        prompt += "[" + r.agent_name + "]: " + r.output + "\n";
    }

    prompt += "\nProvide a final synthesized answer:";

    SamplingParams params;
    params.temperature = 0.5f;
    params.max_tokens = 1024;
    return core_model_.generate(prompt, params);
}

std::string Supervisor::process(const std::string &query)
{
    auto tasks = plan_tasks(query);

    if (tasks.empty()) {
        // No delegation needed; answer directly.
        SamplingParams params;
        params.temperature = 0.7f;
        params.max_tokens = 1024;
        return core_model_.generate(query, params);
    }

    auto results = execute_tasks(tasks);
    return synthesize(query, results);
}

}  // namespace agent
}  // namespace serius