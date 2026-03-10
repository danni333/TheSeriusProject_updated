#pragma once

#include <cstdint>
#include <string>

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
    bool running_ = false;
};

}  // namespace server
}  // namespace serius
