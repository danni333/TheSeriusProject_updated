#include "latency.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "../infer/llama_inference.h"
#include "../infer/sampler.h"
#include "../io/config_loader.h"
#include "../io/gguf_loader.h"
#include "../io/hf_llama_loader.h"
#include "../tokenizer/bpe.h"
#include "../tokenizer/vocab.h"
#include "../transformer/llama_model.h"

namespace fs = std::filesystem;

namespace serius {
namespace bench {

namespace {

bool is_gguf_file(const std::string &path)
{
    return path.size() >= 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
}

/// First .gguf file inside a directory ("" when none).
std::string find_gguf_in_dir(const std::string &dir)
{
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string p = entry.path().string();
        if (is_gguf_file(p)) {
            return p;
        }
    }
    return {};
}

void load_tokenizer(BPETokenizer &tokenizer, const std::string &model_path, const std::string &model_dir)
{
    // Prefer an HF tokenizer.json/vocab.json next to the model.
    std::string vocab_path = model_dir + "/tokenizer.json";
    if (!fs::exists(vocab_path)) {
        vocab_path = model_dir + "/vocab.json";
    }
    if (fs::exists(vocab_path)) {
        try {
            tokenizer.load(vocab_path);
            return;
        }
        catch (const std::exception &e) {
            std::cout << "[Benchmark] Warning: tokenizer load failed (" << e.what() << "), trying GGUF metadata\n";
        }
    }

    // Fall back to the GGUF-embedded tokenizer.
    std::string gguf = is_gguf_file(model_path) ? model_path : find_gguf_in_dir(model_dir);
    if (!gguf.empty()) {
        auto tok = io::load_gguf_tokenizer(gguf);
        if (tok.valid) {
            tokenizer.load_from_gguf(tok);
        }
    }
}

model::LLamaConfig resolve_config(const std::string &model_path, const std::string &model_dir,
                                  const BPETokenizer &tokenizer)
{
    const std::string config_path = model_dir + "/config.json";
    if (fs::exists(config_path)) {
        return io::load_llama_config(config_path);
    }

    // GGUF-only distribution: dimensions come from the file metadata.
    std::string gguf = is_gguf_file(model_path) ? model_path : find_gguf_in_dir(model_dir);
    model::LLamaConfig cfg{};
    if (!gguf.empty()) {
        const auto mc = io::gguf_to_model_config(gguf);
        if (mc.vocab_size > 0) {
            cfg.vocab_size = mc.vocab_size;
            cfg.hidden_size = mc.d_model;
            cfg.num_hidden_layers = mc.n_layers;
            cfg.num_attention_heads = mc.n_heads;
            cfg.num_key_value_heads = mc.n_kv_heads;
            cfg.intermediate_size = mc.d_ff;
            cfg.max_position_embeddings = mc.max_seq_len;
            cfg.rope_theta = mc.rope_theta;
            cfg.rms_norm_eps = mc.rms_norm_eps;
        }
    }
    if (cfg.vocab_size == 0) {
        cfg.vocab_size = tokenizer.vocab_size() > 0 ? tokenizer.vocab_size() : 32000;
    }
    return cfg;
}

bool load_weights(model::LLaMAModel &model, const std::string &model_path, const std::string &model_dir)
{
    if (is_gguf_file(model_path) && fs::is_regular_file(model_path)) {
        return io::load_llama_weights_from_gguf_file(model_path, &model);
    }
    if (fs::is_directory(model_dir)) {
        return io::load_llama_weights_from_dir(model_dir, &model);
    }
    if (fs::exists(model_dir)) {
        return io::load_llama_weights_from_file(model_dir, &model);
    }
    return false;
}

}  // namespace

std::unique_ptr<BenchModel> load_bench_model(const std::string &model_path, bool allow_gpu)
{
    auto bm = std::make_unique<BenchModel>();
    bm->tokenizer = std::make_unique<BPETokenizer>();

    std::string model_dir = model_path;
    if (!fs::is_directory(model_path)) {
        const size_t pos = model_path.find_last_of("/\\");
        model_dir = (pos != std::string::npos) ? model_path.substr(0, pos) : ".";
    }

    load_tokenizer(*bm->tokenizer, model_path, model_dir);
    if (bm->tokenizer->vocab_size() <= 0) {
        std::cout << "[Benchmark] Warning: no tokenizer available — decode output "
                     "will not be checked for EOS\n";
    }

    const model::LLamaConfig cfg = resolve_config(model_path, model_dir, *bm->tokenizer);
    bm->model = std::make_unique<model::LLaMAModel>(cfg);

    if (!load_weights(*bm->model, model_path, model_dir)) {
        std::cout << "[Benchmark] No usable weights found for " << model_path << "\n";
        return nullptr;
    }

    if (allow_gpu) {
        const auto dec = infer::resolve_llama_gpu(*bm->model, infer::LlamaGpuMode::kAuto);
        if (dec.use_gpu) {
            try {
                if (infer::offload_llama_model_to_gpu(*bm->model)) {
                    std::cout << "[Benchmark] Model offloaded to GPU (" << dec.reason << ")\n";
                }
            }
            catch (const std::exception &e) {
                std::cout << "[Benchmark] GPU offload failed: " << e.what() << " — staying on CPU\n";
            }
        }
    }
    return bm;
}

LatencyResult bench_decode(const BenchModel &bm, const std::string &prompt, int max_tokens)
{
    LatencyResult result;
    BPETokenizer &tokenizer = *bm.tokenizer;
    model::LLaMAModel &model = *bm.model;

    std::vector<serius::TokenId> tokens = tokenizer.encode(prompt);
    result.prompt_tokens = static_cast<int>(tokens.size());
    if (result.prompt_tokens == 0 || max_tokens <= 0) {
        return result;
    }

    infer::KVCache cache;
    infer::Sampler sampler;

    core::Tensor input({tokens.size()});
    for (size_t i = 0; i < tokens.size(); ++i) {
        input[i] = static_cast<float>(tokens[i]);
    }

    auto t_prefill0 = std::chrono::high_resolution_clock::now();
    core::Tensor logits = infer::llama_forward_inference_last(model, input, &cache);
    auto t_prefill1 = std::chrono::high_resolution_clock::now();

    if (logits.ndim() == 0 || logits.size() == 0) {
        return result;
    }

    result.ttft_ms = std::chrono::duration<float, std::milli>(t_prefill1 - t_prefill0).count();
    const double prefill_s = std::chrono::duration<double>(t_prefill1 - t_prefill0).count();
    result.prefill_tps = prefill_s > 0.0 ? result.prompt_tokens / static_cast<float>(prefill_s) : 0.0f;

    int next_id = sampler.sample(logits);
    size_t pos = tokens.size();
    double decode_s = 0.0;

    for (int step = 0; step < max_tokens; ++step) {
        const std::string tok_str = tokenizer.decode({static_cast<serius::TokenId>(next_id)});
        if (next_id == static_cast<int>(tokenizer.eos_id()) || tok_str == "<|im_end|>") {
            break;
        }
        ++result.generated_tokens;

        auto t0 = std::chrono::high_resolution_clock::now();
        logits = infer::llama_forward_single_token(model, next_id, pos, &cache);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (logits.ndim() == 0 || logits.size() == 0) {
            break;
        }
        decode_s += std::chrono::duration<double>(t1 - t0).count();
        ++pos;
        next_id = sampler.sample(logits);
    }

    result.decode_tps = decode_s > 0.0 ? static_cast<float>(result.generated_tokens / decode_s) : 0.0f;
    result.total_ms = result.ttft_ms + static_cast<float>(decode_s * 1000.0);
    return result;
}

LatencyResult bench_single(const std::string &model_path, const std::string &prompt, int max_tokens)
{
    LatencyResult result;

    try {
        auto bm = load_bench_model(model_path, /*allow_gpu=*/true);
        if (!bm) {
            std::cout << "[Benchmark] No usable model — metrics left at 0 (no simulated values)\n";
            return result;
        }
        result = bench_decode(*bm, prompt, max_tokens);

        std::cout << "[Benchmark] Results (real inference):\n";
        std::cout << "  Prompt tokens: " << result.prompt_tokens << "\n";
        std::cout << "  Generated tokens: " << result.generated_tokens << "\n";
        std::cout << "  TTFT: " << result.ttft_ms << " ms\n";
        std::cout << "  Prefill TPS: " << result.prefill_tps << " tok/s\n";
        std::cout << "  Decode TPS: " << result.decode_tps << " tok/s\n";
        std::cout << "  Total time: " << result.total_ms << " ms\n";
    }
    catch (const std::exception &e) {
        std::cout << "[Benchmark] Error: " << e.what() << "\n";
        result.ttft_ms = 0.0f;
        result.decode_tps = 0.0f;
        result.prefill_tps = 0.0f;
        result.generated_tokens = 0;
        result.total_ms = 0.0f;
    }

    return result;
}

}  // namespace bench
}  // namespace serius
