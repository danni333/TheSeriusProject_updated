#include "llm_client.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace {

// Split "scheme://authority/path" into client parameters. Returns false on
// unsupported schemes or missing host.
bool split_url(const std::string &url, bool &ssl, std::string &host, int &port,
               std::string &path)
{
    const size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) { return false; }
    const std::string scheme = url.substr(0, scheme_end);
    ssl = scheme == "https";
    if (!ssl && scheme != "http") { return false; }

    const size_t path_start = url.find('/', scheme_end + 3);
    std::string authority = url.substr(scheme_end + 3,
                                       path_start == std::string::npos
                                           ? std::string::npos
                                           : path_start - (scheme_end + 3));
    path = path_start == std::string::npos ? "/" : url.substr(path_start);

    // Strip any userinfo component.
    const size_t at = authority.rfind('@');
    if (at != std::string::npos) { authority = authority.substr(at + 1); }

    const size_t colon = authority.rfind(':');
    if (colon != std::string::npos) {
        host = authority.substr(0, colon);
        const std::string port_str = authority.substr(colon + 1);
        // Strict parse: atoi("80abc") silently yields 80. Require all digits.
        if (port_str.empty() ||
            !std::all_of(port_str.begin(), port_str.end(),
                         [](unsigned char c) { return std::isdigit(c) != 0; })) {
            return false;
        }
        const long p = std::strtol(port_str.c_str(), nullptr, 10);
        if (p <= 0 || p > 65535) {
            return false;
        }
        port = static_cast<int>(p);
    } else {
        host = authority;
        port = ssl ? 443 : 80;
    }
    return !host.empty() && port > 0;
}

template <typename Client>
std::string post_json(Client &cli, const std::string &path, const httplib::Headers &headers,
                      const std::string &body)
{
    auto res = cli.Post(path, headers, body, "application/json");
    if (!res) { return ""; }
    return res->body;
}

}  // namespace

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

    // No code fence: returning the whole response would feed prose into the
    // compiler (guaranteed wasted compile cycle, or a wrong block if the
    // prose contains stray backticks). Empty signals extraction failure.
    return "";
}

std::string LLMClient::http_post(const std::string &url, const std::string &json_body) const
{
    // Direct HTTP via cpp-httplib: no shell, no temp files, and the URL /
    // API key are passed as data — never interpolated into a command line.
    bool ssl = false;
    std::string host, path;
    int port = 0;
    if (!split_url(url, ssl, host, port, path)) {
        return "";
    }

    httplib::Headers headers = {{"Content-Type", "application/json"}};
    if (!config_.api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + config_.api_key);
    }

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (ssl) {
        httplib::SSLClient cli(host, port);
        cli.set_connection_timeout(10);
        cli.set_read_timeout(120);
        return post_json(cli, path, headers, json_body);
    }
#endif
    if (ssl) {
        // HTTPS requires an OpenSSL-enabled httplib build.
        return "";
    }

    httplib::Client cli(host, port);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(120);
    return post_json(cli, path, headers, json_body);
}

std::string LLMClient::generate(const std::string &prompt) const
{
    // Build request body based on API type. nlohmann/json handles escaping,
    // so prompts containing quotes/newlines can't corrupt the request.
    const bool is_ollama =
        config_.api_endpoint.find("ollama") != std::string::npos ||
        config_.api_endpoint.find("localhost:11434") != std::string::npos;

    nlohmann::json body;
    if (is_ollama) {
        // Ollama format
        body["model"] = config_.model_name;
        body["prompt"] = prompt;
        body["stream"] = false;
        body["options"] = nlohmann::json{{"temperature", config_.temperature}};
    } else {
        // OpenAI-compatible format
        body["model"] = config_.model_name;
        body["messages"] = nlohmann::json::array(
            {nlohmann::json{{"role", "user"}, {"content", prompt}}});
        body["temperature"] = config_.temperature;
        body["max_tokens"] = config_.max_tokens;
    }

    std::string response = http_post(config_.api_endpoint, body.dump());

    if (response.empty()) {
        return "";
    }

    // Parse the JSON response properly: Ollama returns {"response": "..."},
    // OpenAI-compatible returns {"choices": [{"message": {"content": ...}}]}.
    const auto j = nlohmann::json::parse(response, nullptr, false);
    if (j.is_discarded()) {
        return response;  // Not JSON — return raw
    }

    if (j.contains("response") && j["response"].is_string()) {
        return j["response"].get<std::string>();
    }

    if (j.contains("choices") && j["choices"].is_array() &&
        !j["choices"].empty()) {
        const auto &choice = j["choices"][0];
        if (choice.contains("message") && choice["message"].is_object() &&
            choice["message"].contains("content") &&
            choice["message"]["content"].is_string()) {
            return choice["message"]["content"].get<std::string>();
        }
        if (choice.contains("text") && choice["text"].is_string()) {
            return choice["text"].get<std::string>();
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
