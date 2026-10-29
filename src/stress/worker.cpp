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
                           int max_tokens)
    : server_url_(server_url), prompt_(prompt), max_tokens_(max_tokens) {}

void StressWorker::run(std::atomic<bool>& running, StressResult& result, std::mutex& mtx) {
    httplib::Client cli(server_url_);
    cli.set_connection_timeout(30);
    cli.set_read_timeout(30);

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

        // Small delay between requests
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool StressWorker::send_request(float& latency_ms) {
    httplib::Client cli(server_url_);
    cli.set_connection_timeout(30);
    cli.set_read_timeout(30);

    nlohmann::json body;
    body["model"] = "serius";
    body["messages"] = {{{"role", "user"}, {"content", prompt_}}};
    body["max_tokens"] = max_tokens_;
    body["stream"] = false;

    auto start = std::chrono::steady_clock::now();
    auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    auto end = std::chrono::steady_clock::now();

    latency_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return res && res->status == 200;
}

}  // namespace stress
}  // namespace serius
