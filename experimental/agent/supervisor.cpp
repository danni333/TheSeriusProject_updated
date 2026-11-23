#include "supervisor.h"
#include "specialist.h"

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

    // Minimal JSON parsing: extract agent names and prompts.
    // Look for "agent" and "prompt" keys.
    size_t pos = 0;
    int task_id = 0;
    while ((pos = plan.find("\"agent\"", pos)) != std::string::npos) {
        Task t;
        t.id = std::to_string(task_id++);

        // Extract agent name.
        size_t colon = plan.find(':', pos);
        size_t q1 = plan.find('"', colon + 1);
        size_t q2 = plan.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) {
            t.agent_name = plan.substr(q1 + 1, q2 - q1 - 1);
        }

        // Extract prompt.
        size_t prompt_pos = plan.find("\"prompt\"", q2);
        if (prompt_pos != std::string::npos) {
            size_t pcolon = plan.find(':', prompt_pos);
            size_t pq1 = plan.find('"', pcolon + 1);
            size_t pq2 = plan.find('"', pq1 + 1);
            if (pq1 != std::string::npos && pq2 != std::string::npos) {
                t.prompt = plan.substr(pq1 + 1, pq2 - pq1 - 1);
            }
        }

        if (!t.agent_name.empty()) {
            tasks.push_back(std::move(t));
        }
        pos = q2;
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