#include "specialist.h"

#include <algorithm>
#include <cctype>

namespace serius {
namespace agent {

Specialist::Specialist(std::string name, std::string specialty, Model *model)
    : name_(std::move(name)), specialty_(std::move(specialty)), model_(model)
{
}

std::string Specialist::execute(const std::string &task) const
{
    if (!model_) {
        return "Error: no model available for specialist " + name_;
    }

    std::string system_prompt = "You are a " + name_ + ". Your specialty is " + specialty_ +
                                ". Provide a concise, accurate response.";
    SamplingParams params;
    params.temperature = 0.7f;
    params.max_tokens = 512;
    return model_->generate(system_prompt + "\nTask: " + task, params);
}

static std::string to_lower(const std::string &s)
{
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

bool Specialist::can_handle(const std::string &keyword) const
{
    std::string lower_kw = to_lower(keyword);
    std::string lower_spec = to_lower(specialty_);
    return lower_spec.find(lower_kw) != std::string::npos;
}

}  // namespace agent
}  // namespace serius