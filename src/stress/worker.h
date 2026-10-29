#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <vector>

namespace serius {
namespace stress {

struct StressResult;

/// Worker thread that simulates a single user making requests to the server.
class StressWorker {
public:
    StressWorker(const std::string& server_url,
                 const std::string& prompt,
                 int max_tokens);

    /// Run the worker until stopped
    void run(std::atomic<bool>& running, StressResult& result, std::mutex& mtx);

private:
    std::string server_url_;
    std::string prompt_;
    int max_tokens_;

    /// Send a single HTTP request to the server
    bool send_request(float& latency_ms);
};

}  // namespace stress
}  // namespace serius
