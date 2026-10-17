#pragma once

#include <cstdint>
#include <string>

#include <httplib.h>

namespace serius {
namespace server {

/// Minimal HTTP server for inference API (OpenAI-compatible endpoints).
/// Uses cpp-httplib; run() blocks until shutdown.
class Server
{
public:
    explicit Server(uint16_t port = 8080);
    ~Server() = default;

    /// Start listening; blocks until stop or error.
    /// Returns 0 on normal stop, non-zero on error.
    int run();

    /// Stop the server (thread-safe signal to exit).
    void stop();

    uint16_t port() const { return port_; }

private:
    uint16_t port_;
    httplib::Server server_;
};

/// Build a small OpenAI-compatible completion payload from the last user prompt.
std::string build_completion_text(const std::string &prompt);

/// Extract the last user message from an OpenAI-like chat JSON payload.
std::string extract_last_user_message(const std::string &json_body);

}  // namespace server
}  // namespace serius
