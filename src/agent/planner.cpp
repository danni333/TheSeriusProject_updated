#include "planner.h"

#include <algorithm>
#include <sstream>

namespace serius {
namespace agent {

Planner::Planner(Model &model) : model_(model)
{
}

PlanResult Planner::decompose(const std::string &query) const
{
    std::string system =
        "You are a task planner. Break the user's request into independent sub-tasks.\n"
        "Respond with a numbered list, one task per line, prefixed with '- '.\n"
        "Each task should be a concise, actionable instruction.\n"
        "If the request is already atomic, respond with a single task.";

    return decompose(query, system);
}

PlanResult Planner::decompose(const std::string &query, const std::string &system_prompt) const
{
    PlanResult result;

    std::string full_prompt = system_prompt + "\n\nRequest: " + query + "\n\nSub-tasks:";

    SamplingParams params;
    params.temperature = 0.3f;
    params.max_tokens = 512;
    std::string plan = model_.generate(full_prompt, params);

    result.reasoning = plan;

    // Parse lines starting with "- " as tasks.
    std::istringstream iss(plan);
    std::string line;
    while (std::getline(iss, line)) {
        // Trim whitespace.
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t");
        std::string trimmed = line.substr(start, end - start + 1);

        if (trimmed.size() >= 2 && trimmed[0] == '-' && trimmed[1] == ' ') {
            SubTask task;
            task.description = trimmed.substr(2);
            // Extract keywords: words longer than 3 chars.
            std::istringstream desc(task.description);
            std::string word;
            while (desc >> word) {
                if (word.size() > 3) {
                    task.keywords.push_back(word);
                }
            }
            result.tasks.push_back(std::move(task));
        }
    }

    // If no tasks parsed, treat the whole query as a single task.
    if (result.tasks.empty()) {
        SubTask fallback;
        fallback.description = query;
        result.tasks.push_back(std::move(fallback));
    }

    return result;
}

}  // namespace agent
}  // namespace serius