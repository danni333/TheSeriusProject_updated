#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace serius {
namespace bench {

struct LatencyResult
{
    float ttft_ms = 0.0f;           // Time to first token
    float decode_tps = 0.0f;        // Tokens/sec during generation
    float prefill_tps = 0.0f;       // Tokens/sec during prompt processing
    int prompt_tokens = 0;
    int generated_tokens = 0;
    float total_ms = 0.0f;
};

/// Benchmark single-user latency metrics.
/// Loads the model, encodes the prompt, and measures:
/// - Time to first token (TTFT)
/// - Tokens per second during prefill
/// - Tokens per second during decode
LatencyResult bench_single(const std::string &model_path,
                          const std::string &prompt,
                          int max_tokens = 256);

}  // namespace bench
}  // namespace serius
