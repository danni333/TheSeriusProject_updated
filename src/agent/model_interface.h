#pragma once

#include <string>
#include <vector>

namespace serius {
namespace agent {

/// Sampling parameters passed to the model.
struct SamplingParams
{
    float temperature = 0.7f;
    int max_tokens = 512;
    std::vector<std::string> stop_tokens;
};

/// Language model interface used by the ReAct engine.
class Model
{
   public:
    virtual ~Model() = default;
    virtual std::string generate(const std::string &prompt,
                                 const SamplingParams &params = {}) = 0;
};

/// Parsed [ACTION] block from the model output.
struct ActionCall
{
    std::string name;
    std::string args_json;
};

}  // namespace agent
}  // namespace serius
