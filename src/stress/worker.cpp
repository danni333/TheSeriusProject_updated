#include "worker.h"
#include "stress_runner.h"
#include <thread>
#include <chrono>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace serius {
namespace stress {

StressWorker::StressWorker(const std::string& server_url,
                           const std::string& prompt,
                           int max_tokens,
                           const std::string& model_name,
                           const std::string& endpoint,
                           int timeout_seconds,
                           int interval_ms)
    : server_url_(server_url), prompt_(prompt), max_tokens_(max_tokens),
      model_name_(model_name), endpoint_(endpoint),
      timeout_seconds_(timeout_seconds), interval_ms_(interval_ms) {}

void StressWorker::run(std::atomic<bool>& running, StressResult& result, std::mutex& mtx) {
    while (running) {
        float latency_ms = 0.0f;
        bool success = send_request(latency_ms);

        {
            std::lock_guard<std::mutex> lock(mtx);
            result.total_requests++;
            if (success) {
                result.latencies_ms.push_back(latency_ms);
            } else {
                result.total_errors++;
            }
        }

        // Delay between requests; a 0 interval still yields the thread.
        if (interval_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        } else {
            std::this_thread::yield();
        }
    }
}

bool StressWorker::send_request(float& latency_ms) {
    httplib::Client cli(server_url_);
    cli.set_connection_timeout(timeout_seconds_);
    cli.set_read_timeout(timeout_seconds_);

    nlohmann::json body;
    body["model"] = model_name_;
    body["messages"] = {{{"role", "user"}, {"content", prompt_}}};
    body["max_tokens"] = max_tokens_;
    body["stream"] = false;

    auto start = std::chrono::steady_clock::now();
    auto res = cli.Post(endpoint_, body.dump(), "application/json");
    auto end = std::chrono::steady_clock::now();

    latency_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return res && res->status == 200;
}

}  // namespace stress
}  // namespace serius
