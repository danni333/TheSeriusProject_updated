#include "llm_client.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#define popen _popen
#define pclose _pclose
#endif

// For HTTP requests, we'll use a simple approach with system curl
// In production, consider using libcurl or cpp-httplib

namespace serius {
namespace autoopt {

LLMClient::LLMClient(const LLMConfig &config) : config_(config)
{
}

std::string LLMClient::build_optimization_prompt(const std::string &kernel_name,
                                                 const std::string &current_source,
                                                 const std::string &performance_info) const
{
    std::ostringstream prompt;
    prompt << "You are an expert C++ performance optimization engineer.\n\n";
    prompt << "Task: Optimize the following kernel for performance.\n";
    prompt << "Kernel name: " << kernel_name << "\n\n";
    prompt << "Performance info:\n" << performance_info << "\n\n";
    prompt << "Current implementation:\n```cpp\n" << current_source << "\n```\n\n";
    prompt << "Requirements:\n";
    prompt << "1. Maintain numerical correctness (do not change semantics)\n";
    prompt << "2. Optimize for CPU performance (consider SIMD, cache locality, loop unrolling)\n";
    prompt << "3. Keep the function signature unchanged\n";
    prompt << "4. Return ONLY the optimized C++ code (no explanations, no markdown)\n\n";
    prompt << "Optimized code:\n";

    return prompt.str();
}

std::string LLMClient::extract_code_from_response(const std::string &response) const
{
    // Try to extract code from markdown code blocks
    size_t code_start = response.find("```cpp");
    if (code_start != std::string::npos) {
        code_start = response.find('\n', code_start) + 1;
        size_t code_end = response.find("```", code_start);
        if (code_end != std::string::npos) {
            return response.substr(code_start, code_end - code_start);
        }
    }

    // Try generic code block
    code_start = response.find("```");
    if (code_start != std::string::npos) {
        code_start = response.find('\n', code_start) + 1;
        size_t code_end = response.find("```", code_start);
        if (code_end != std::string::npos) {
            return response.substr(code_start, code_end - code_start);
        }
    }

    // If no code blocks, return the whole response
    return response;
}

std::string LLMClient::http_post(const std::string &url, const std::string &json_body) const
{
    // Write JSON body to temp file
    std::string temp_file = "llm_request_" + std::to_string(std::hash<std::string>{}(json_body)) + ".json";
    std::ofstream out(temp_file);
    if (!out) {
        return "";
    }
    out << json_body;
    out.close();

    // Build curl command
    std::ostringstream cmd;
    cmd << "curl -s -X POST \"" << url << "\"";
    cmd << " -H \"Content-Type: application/json\"";

    if (!config_.api_key.empty()) {
        cmd << " -H \"Authorization: Bearer " << config_.api_key << "\"";
    }

    cmd << " -d @" << temp_file;

    // Execute curl
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        std::remove(temp_file.c_str());
        return "";
    }

    // Read response
    char buffer[4096];
    std::string response;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        response += buffer;
    }
    pclose(pipe);

    // Clean up temp file
    std::remove(temp_file.c_str());

    return response;
}

std::string LLMClient::generate(const std::string &prompt) const
{
    // Build request body based on API type
    std::ostringstream json;
    json << "{";

    // Check if this is Ollama (default) or OpenAI-compatible
    if (config_.api_endpoint.find("ollama") != std::string::npos ||
        config_.api_endpoint.find("localhost:11434") != std::string::npos) {
        // Ollama format
        json << "\"model\": \"" << config_.model_name << "\",";
        json << "\"prompt\": \"" << prompt << "\",";
        json << "\"stream\": false,";
        json << "\"options\": {\"temperature\": " << config_.temperature << "}";
    } else {
        // OpenAI-compatible format
        json << "\"model\": \"" << config_.model_name << "\",";
        json << "\"messages\": [{\"role\": \"user\", \"content\": \"" << prompt << "\"}],";
        json << "\"temperature\": " << config_.temperature << ",";
        json << "\"max_tokens\": " << config_.max_tokens;
    }

    json << "}";

    std::string response = http_post(config_.api_endpoint, json.str());

    if (response.empty()) {
        return "";
    }

    // Extract text from response (simplified parsing)
    // In production, use nlohmann/json for proper parsing
    size_t text_pos = response.find("\"response\"");
    if (text_pos != std::string::npos) {
        // Ollama format
        size_t colon = response.find(":", text_pos);
        size_t start = response.find("\"", colon) + 1;
        size_t end = response.find("\"", start);
        if (end != std::string::npos) {
            return response.substr(start, end - start);
        }
    }

    text_pos = response.find("\"content\"");
    if (text_pos != std::string::npos) {
        // OpenAI format
        size_t colon = response.find(":", text_pos);
        size_t start = response.find("\"", colon) + 1;
        size_t end = response.find("\"", start);
        if (end != std::string::npos) {
            return response.substr(start, end - start);
        }
    }

    // Fallback: return raw response
    return response;
}

std::string LLMClient::generate_optimized_code(const std::string &kernel_name,
                                               const std::string &current_source,
                                               const std::string &performance_info) const
{
    std::string prompt = build_optimization_prompt(kernel_name, current_source, performance_info);
    std::string response = generate(prompt);

    if (response.empty()) {
        return "";
    }

    return extract_code_from_response(response);
}

}  // namespace autoopt
}  // namespace serius
