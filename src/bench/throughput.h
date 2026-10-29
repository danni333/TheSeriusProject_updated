#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>

namespace serius {
namespace bench {

struct ThroughputResult
{
    float requests_per_sec = 0.0f;
    float total_tokens_per_sec = 0.0f;
    float avg_latency_ms = 0.0f;
    float p99_latency_ms = 0.0f;
    int concurrent_users = 0;
    int total_requests = 0;
    int total_tokens = 0;
};

/// Benchmark multi-user throughput metrics.
/// Launches N concurrent workers, each sending multiple requests.
/// Measures aggregate throughput and per-request latency statistics.
ThroughputResult bench_throughput(const std::string &model_path,
                                  const std::vector<std::string> &prompts,
                                  int concurrent_users,
                                  int requests_per_user,
                                  int max_tokens = 256);

}  // namespace bench
}  // namespace serius
