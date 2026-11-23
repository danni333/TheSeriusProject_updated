#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "../tokenizer/bpe.h"
#include "../transformer/llama_model.h"

namespace serius {

namespace bench {

struct LatencyResult
{
    float ttft_ms = 0.0f;      // Time to first token
    float decode_tps = 0.0f;   // Tokens/sec during generation
    float prefill_tps = 0.0f;  // Tokens/sec during prompt processing
    int prompt_tokens = 0;
    int generated_tokens = 0;
    float total_ms = 0.0f;
};

/// A fully loaded model + tokenizer, reusable across benchmark requests.
/// Loading is the expensive part; bench_decode only measures inference.
struct BenchModel
{
    std::unique_ptr<model::LLaMAModel> model;
    std::unique_ptr<BPETokenizer> tokenizer;
};

/// Load a model for benchmarking. Accepts the same inputs as `serius chat`:
/// a .gguf file, a model directory (config.json + safetensors/GGUF), or
/// hf://repo_id is NOT supported here (resolve via `serius import` first).
/// GPU offload follows the same auto policy as the CLI (SERIUS_NO_GPU=1
/// forces CPU). Returns nullptr when weights cannot be loaded.
std::unique_ptr<BenchModel> load_bench_model(const std::string &model_path, bool allow_gpu = true);

/// Run one prompt through prefill + decode on an already-loaded model and
/// fill `result` with real measured metrics. Uses a fresh KV cache per call.
LatencyResult bench_decode(const BenchModel &bm, const std::string &prompt, int max_tokens = 256);

/// Benchmark single-user latency metrics.
/// Loads the model, encodes the prompt, and measures:
/// - Time to first token (TTFT)
/// - Tokens per second during prefill
/// - Tokens per second during decode
LatencyResult bench_single(const std::string &model_path, const std::string &prompt, int max_tokens = 256);

}  // namespace bench
}  // namespace serius
