#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "../common/defaults.h"

namespace serius {
namespace stress {

struct StressResult;

/// Worker thread that simulates a single user making requests to the server.
class StressWorker
{
   public:
    StressWorker(const std::string &server_url, const std::string &prompt, int max_tokens,
                 const std::string &model_name = defaults::kModelName,
                 const std::string &endpoint = "/v1/chat/completions",
                 int timeout_seconds = defaults::kStressTimeoutSec, int interval_ms = 100);

    /// Run the worker until stopped
    void run(std::atomic<bool> &running, StressResult &result, std::mutex &mtx);

   private:
    std::string server_url_;
    std::string prompt_;
    int max_tokens_;
    std::string model_name_;
    std::string endpoint_;
    int timeout_seconds_;
    int interval_ms_;

    /// Send a single HTTP request to the server; outputs latency and generated tokens
    bool send_request(float &latency_ms, int &completion_tokens);
};

}  // namespace stress
}  // namespace serius
