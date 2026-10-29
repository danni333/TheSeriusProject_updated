#include "throughput.h"
#include "latency.h"

#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace serius {
namespace bench {

ThroughputResult bench_throughput(const std::string &model_path,
                                  const std::vector<std::string> &prompts,
                                  int concurrent_users,
                                  int requests_per_user,
                                  int max_tokens)
{
    ThroughputResult result;
    result.concurrent_users = concurrent_users;
    result.total_requests = concurrent_users * requests_per_user;

    std::vector<std::thread> workers;
    std::atomic<int> total_tokens{0};
    std::vector<float> latencies;
    std::mutex lat_mutex;
    std::mutex cout_mutex;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < concurrent_users; i++) {
        workers.emplace_back([&, i]() {
            for (int j = 0; j < requests_per_user; j++) {
                // Select prompt cyclically
                size_t prompt_idx = (i * requests_per_user + j) % prompts.size();
                const std::string &prompt = prompts[prompt_idx];

                auto lat_result = bench_single(model_path, prompt, max_tokens);
                total_tokens += lat_result.generated_tokens;

                {
                    std::lock_guard<std::mutex> lock(lat_mutex);
                    latencies.push_back(lat_result.total_ms);
                }
            }
        });
    }

    for (auto &w : workers) {
        w.join();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float total_sec = std::chrono::duration<float>(t1 - t0).count();

    result.total_tokens = total_tokens;

    if (total_sec > 0.0f) {
        result.requests_per_sec = result.total_requests / total_sec;
        result.total_tokens_per_sec = result.total_tokens / total_sec;
    }

    if (!latencies.empty()) {
        std::sort(latencies.begin(), latencies.end());
        result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / latencies.size();
        size_t p99_idx = latencies.size() * 99 / 100;
        result.p99_latency_ms = latencies[p99_idx];
    }

    std::cout << "[Throughput Benchmark] Results:\n";
    std::cout << "  Concurrent users: " << result.concurrent_users << "\n";
    std::cout << "  Total requests: " << result.total_requests << "\n";
    std::cout << "  Total tokens: " << result.total_tokens << "\n";
    std::cout << "  Requests/sec: " << result.requests_per_sec << "\n";
    std::cout << "  Tokens/sec: " << result.total_tokens_per_sec << "\n";
    std::cout << "  Avg latency: " << result.avg_latency_ms << " ms\n";
    std::cout << "  P99 latency: " << result.p99_latency_ms << " ms\n";

    return result;
}

}  // namespace bench
}  // namespace serius
