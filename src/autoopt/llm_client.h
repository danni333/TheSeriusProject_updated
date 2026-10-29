#pragma once

#include <string>
#include <functional>

namespace serius {
namespace autoopt {

/// Configuration for LLM code generation client.
struct LLMConfig
{
    std::string api_endpoint = "http://localhost:11434/api/generate";  // Default: Ollama
    std::string model_name = "codellama";
    float temperature = 0.2f;  // Low temperature for deterministic code
    int max_tokens = 4096;
    std::string api_key;  // Optional: for OpenAI-compatible APIs
};

/// Simple LLM client for code generation.
/// Supports OpenAI-compatible APIs (including Ollama local).
class LLMClient
{
   public:
    explicit LLMClient(const LLMConfig &config = LLMConfig());

    /// Generate optimized C++ code for a kernel.
    /// @param kernel_name Name of the kernel being optimized
    /// @param current_source Current implementation source code
    /// @param performance_info Performance metrics (latency, bottleneck info)
    /// @return Generated source code (empty on failure)
    std::string generate_optimized_code(const std::string &kernel_name,
                                        const std::string &current_source,
                                        const std::string &performance_info) const;

    /// Generate code with custom prompt.
    std::string generate(const std::string &prompt) const;

    const LLMConfig &config() const { return config_; }
    void set_config(const LLMConfig &config) { config_ = config; }

   private:
    LLMConfig config_;

    /// Build the prompt for code optimization
    std::string build_optimization_prompt(const std::string &kernel_name,
                                          const std::string &current_source,
                                          const std::string &performance_info) const;

    /// Make HTTP POST request to LLM API
    std::string http_post(const std::string &url, const std::string &json_body) const;

    /// Parse LLM response to extract code
    std::string extract_code_from_response(const std::string &response) const;
};

}  // namespace autoopt
}  // namespace serius
