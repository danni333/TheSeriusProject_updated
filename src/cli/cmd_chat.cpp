#include "cli/cmd_chat.h"

#include "cli/args_parser.h"
#include "cli/llama_loader.h"
#ifdef SERIUS_EXPERIMENTAL_GROUNDING
#include "grounding/confidence.h"
#endif
#include "../blas/blas.h"
#include "../common/defaults.h"
#include "../hub/downloader.h"
#include "../hub/search_engine.h"
#include "../safety/red_team.h"
#include "hal/device.h"
#include "hal/fused_kernels.h"
#include "infer/llama_inference.h"
#include "infer/sampler.h"
#include "io/config_loader.h"
#include "io/gguf_loader.h"
#include "io/hf_llama_loader.h"
#include "tokenizer/bpe.h"
#include "transformer/llama_model.h"
#include "transformer/model.h"
#include "transformer/qwen_model.h"
#ifdef SERIUS_USE_CUDA
#include "../blas/matmul_cuda.h"
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace serius {
namespace cli {

namespace {

/// Log-probability of token `id` under softmax(logits[row0]) — numerically
/// stable. `logits` may be {vocab} or {rows, vocab}; row0 selects the row.
float token_logprob(const core::Tensor &logits, size_t row0, int id)
{
    const size_t vocab = logits.shape().back();
    if (id < 0 || static_cast<size_t>(id) >= vocab || logits.size() < vocab) {
        return 0.0f;
    }
    const float *row = logits.data() + row0 * vocab;
    float mx = row[0];
    for (size_t v = 1; v < vocab; ++v) {
        if (row[v] > mx) {
            mx = row[v];
        }
    }
    double sum_exp = 0.0;
    for (size_t v = 0; v < vocab; ++v) {
        sum_exp += std::exp(static_cast<double>(row[v]) - static_cast<double>(mx));
    }
    return static_cast<float>(static_cast<double>(row[static_cast<size_t>(id)]) -
                              (std::log(sum_exp) + static_cast<double>(mx)));
}

}  // namespace

int cmd_chat(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    ArgsParser parser("serius chat", "Interactive chat with the model");

    parser.add_flag({"model", "m", "Path to model directory/config, or hf://repo_id", true, "models/smollm2-135m"});
    parser.add_flag({"quant", "", "GGUF quant tag for hf:// models (e.g. q4_k_m); default: auto", true, ""});
    parser.add_flag({"type", "", "Model type: 'llama', 'qwen', or 'transformer'", true, "llama"});
    parser.add_flag({"temperature", "", "Sampling temperature", true, "0.7"});
    parser.add_flag({"top-k", "", "Top-k sampling (0 = disabled)", true, "40"});
    parser.add_flag({"top-p", "", "Top-p (nucleus) sampling", true, "0.9"});
    parser.add_flag({"max-tokens", "", "Maximum tokens to generate", true, "128"});
    parser.add_flag({"seed", "", "Random seed for sampling (0 = random)", true, "0"});
    parser.add_flag({"repeat-penalty", "", "Repetition penalty", true, "1.15"});
    parser.add_flag({"prompt", "p", "Single prompt to evaluate non-interactively", true, ""});
    parser.add_flag({"system", "s", "System prompt", true, "You are a helpful AI assistant."});
    parser.add_flag({"raw", "", "Do not format prompt with ChatML template", false, ""});
    parser.add_flag({"gpu", "", "GPU backend (auto/cpu/cuda)", true, "auto"});
    parser.add_flag({"gpu-layers", "", "Partial GPU offload: layers on GPU, rest CPU (-1 = auto)", true, "-1"});
    parser.add_flag(
        {"embed-model", "", "Embedding model (GGUF/safetensors dir) enabling semantic scorers (--toxicity)", true, ""});
    parser.add_flag(
        {"toxicity", "", "Score each response for toxicity (0-1); warns above --toxicity-threshold", false, ""});
    parser.add_flag({"toxicity-threshold", "", "Toxicity warning threshold", true, "0.7"});

    if (!parser.parse(argc, argv)) {
        std::cerr << "Error: " << parser.error() << "\n";
        parser.print_help();
        return 1;
    }

    if (parser.has("help")) {
        parser.print_help();
        return 0;
    }

    std::string model_path = parser.get("model");
    std::string model_type = parser.get("type", "llama");
    float temperature = parser.get_float("temperature", defaults::kTemperature);
    int top_k = parser.get_int("top-k", defaults::kTopK);
    float top_p = parser.get_float("top-p", defaults::kTopP);
    int max_new_tokens = parser.get_int("max-tokens", 128);
    uint64_t seed = std::strtoull(parser.get("seed", "0").c_str(), nullptr, 10);
    float repeat_penalty = parser.get_float("repeat-penalty", 1.15f);
    std::string single_prompt = parser.get("prompt");
    std::string system_prompt = parser.get("system", "You are a helpful AI assistant.");
    bool raw_mode = parser.has("raw");
    const std::string gpu_mode = parser.get("gpu", "auto");

    // If seed is 0, use random seed
    if (seed == 0) {
        std::random_device rd;
        seed = rd();
    }

    // Qwen2-family models (GGUF arch "qwen2", HF config.json) run through
    // the LLaMA path: architecture specifics come from config/metadata.
    if (model_type == "qwen") {
        model_type = "llama";
    }
    if (model_type != "llama") {
        std::cerr << "Model type '" << model_type << "' not yet fully wired in CLI chat.\n";
        return 1;
    }

    // Shared loader: hf:// resolution, tokenizer, config, weights, GPU offload.
    auto loaded = load_llama_cli_model(model_path, gpu_mode, parser.get("quant"), parser.get_int("gpu-layers", -1));
    if (!loaded) {
        return 1;
    }
    model::LLaMAModel &model = *loaded->model;
    BPETokenizer &tokenizer = loaded->tokenizer;

    bool is_chatml = (tokenizer.get_id("<|im_start|>") != BPETokenizer::UNK_ID);

    infer::Sampler sampler(temperature, top_k, top_p, seed);

    // Optional post-generation toxicity check. Without --embed-model the
    // classifier uses its lexical keyword path; with a real embedding model
    // it also scores prototype similarity semantically.
    const bool toxicity_enabled = parser.has("toxicity");
    const float toxicity_threshold = parser.get_float("toxicity-threshold", 0.7f);
    safety::ToxicityClassifier toxicity_classifier;
    std::unique_ptr<hub::EmbeddingManager> embed_mgr;
    const std::string embed_model = parser.get("embed-model");
    if (!embed_model.empty()) {
        embed_mgr = std::make_unique<hub::EmbeddingManager>();
        embed_mgr->load_model(embed_model);
        if (embed_mgr->is_model_loaded()) {
            toxicity_classifier.set_embedder(embed_mgr->document_embed_fn());
            std::cerr << "[chat] embed model loaded: " << embed_model << "\n";
        } else {
            std::cerr << "[chat] embed model unavailable, lexical scorers only\n";
        }
    }

    // Reuse one KVCache across turns: the GPU arena + per-layer views
    // persist, so the CUDA decode graph (bound to those raw pointers)
    // stays valid instead of being recaptured every turn.
    infer::KVCache cache;

#ifdef SERIUS_USE_CUDA
    if (std::getenv("SERIUS_NO_GPU") == nullptr && model.embed_tokens() && model.embed_tokens()->data.is_gpu()) {
        // Decode-graph warmup on the persistent cache: captures the CUDA
        // graph once here so the first real turn doesn't pay capture
        // cost inside its first decode step.
        const size_t wn = 16;
        core::Tensor warm_in({wn});
        for (size_t i = 0; i < wn; ++i) {
            warm_in[i] = static_cast<float>(1000 + (i % 1000));
        }
        (void)infer::llama_forward_inference_last(model, warm_in, &cache);
        (void)infer::llama_forward_single_token(model, 1000, wn, &cache);
        (void)infer::llama_forward_single_token(model, 1001, wn + 1, &cache);
        cache.clear();
    }
#endif

    auto run_turn = [&](const std::string &user_input) {
        std::string prompt_text = user_input;
        if (is_chatml && !raw_mode) {
            prompt_text = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n" + user_input +
                          "<|im_end|>\n<|im_start|>assistant\n";
        }

        std::vector<TokenId> ids = tokenizer.encode(prompt_text);
        {
            const char *_prof = std::getenv("SERIUS_PROF");
            if (_prof != nullptr && _prof[0] != '\0') {
                std::fprintf(stderr, "[chatprof] prompt_tokens=%zu\n", ids.size());
            }
        }
        if (!is_chatml && (ids.empty() || ids[0] != tokenizer.bos_id())) {
            ids.insert(ids.begin(), tokenizer.bos_id());
        }

        cache.clear();
        core::Tensor prompt_input({ids.size()});
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_input[i] = static_cast<float>(ids[i]);
        }

        std::vector<int> recent_ids;
        recent_ids.reserve(ids.size() + max_new_tokens);
        for (TokenId tid : ids) recent_ids.push_back(static_cast<int>(tid));
        // Dense occurrence counts for the repetition penalty: updated
        // incrementally so the per-token sampler does O(1) lookups instead
        // of rebuilding a hash map over the whole history.
        thread_local std::vector<int> tok_counts;
        if (tok_counts.size() < tokenizer.vocab_size()) {
            tok_counts.assign(tokenizer.vocab_size(), 0);
        } else {
            std::fill(tok_counts.begin(), tok_counts.end(), 0);
        }
        for (int tid : recent_ids) {
            if (tid >= 0 && static_cast<size_t>(tid) < tok_counts.size()) {
                ++tok_counts[tid];
            }
        }

        auto gen_start = std::chrono::steady_clock::now();
        core::Tensor logits = infer::llama_forward_inference_last(model, prompt_input, &cache);
        const double t_prefill = std::chrono::duration<double>(std::chrono::steady_clock::now() - gen_start).count();

        // GPU top-k preselection + CPU candidate sampling: identical
        // output to Sampler::sample but skips the 152k-element
        // nth_element on the host (~3 ms/token saved).
        auto gpu_sample = [&](const core::Tensor &lg, size_t row0, int &out_id) -> bool {
#ifdef SERIUS_USE_CUDA
            const void *gp = lg.raw_data_gpu();
            if (gp == nullptr) {
                return false;
            }
            const size_t vs = lg.shape().back();
            const float *lgp = static_cast<const float *>(gp) + row0 * vs;
            thread_local std::vector<int> cidx;
            thread_local std::vector<float> cval;
            // Kernels overwrite the first `cnt` entries only — resize once.
            if (cidx.size() < 16384) {
                cidx.resize(16384);
            }
            if (cval.size() < 16384) {
                cval.resize(16384);
            }
            int cnt = hal::gpu_topk_raw_candidates(lgp, static_cast<int>(vs), 16384, 2048, cidx.data(), cval.data());
            if (cnt <= 0) {
                // Cooperative path unavailable/failed — iterative fallback.
                cnt = hal::topk_logits_f32(lgp, static_cast<int>(vs), 384, 16384, cidx.data(), cval.data());
                if (cnt <= 0) {
                    return false;
                }
            }
            if (cnt <= 0) {
                return false;
            }
            cidx.resize(static_cast<size_t>(cnt));
            cval.resize(static_cast<size_t>(cnt));
            const int t =
                sampler.sample_from_candidates_dense(cidx, cval, tok_counts.data(), tok_counts.size(), repeat_penalty);
            if (t < 0) {
                return false;
            }
            out_id = t;
            return true;
#else
            return false;
#endif
        };

        int next_id = 0;
        float pending_lp = 0.0f;
        bool pending_valid = false;
        {
            size_t vocab_size = logits.shape().back();
            size_t last_row = logits.shape()[0] - 1;
            core::Tensor last_logits({vocab_size});
            const float *src = logits.data() + (last_row * vocab_size);
            std::copy(src, src + vocab_size, last_logits.data());
            next_id = sampler.sample(last_logits, recent_ids, repeat_penalty);
            pending_lp = token_logprob(last_logits, 0, next_id);
            pending_valid = true;
        }

        // Per-token log-probs of emitted tokens → grounding confidence.
        std::vector<float> gen_logprobs;
        // Full decoded response → post-generation safety scoring.
        std::string response_text;

        size_t tokens_generated = 0;
        const char *_prof_env = std::getenv("SERIUS_PROF");
        const bool chat_prof = _prof_env != nullptr && _prof_env[0] != '\0';
        double t_dec = 0, t_fwd = 0, t_smp = 0, t_out = 0;
        auto now_s = [] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        };
        for (int gen = 0; gen < max_new_tokens; ++gen) {
            const double td0 = chat_prof ? now_s() : 0.0;
            std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
            if (chat_prof) {
                t_dec += now_s() - td0;
            }
            // EOS_ID==2 is the GPT-2/LLaMA convention; ChatML models
            // (Qwen, SmolLM) end on <|im_end|>. In the Qwen vocab token 2
            // decodes to "</s>" but is a normal token there, so the legacy
            // stop strings must only apply to non-ChatML models.
            if ((!is_chatml && (next_id == static_cast<int>(BPETokenizer::EOS_ID) || tok_str == "<|endoftext|>" ||
                                tok_str == "</s>")) ||
                tok_str == "<|im_end|>") {
                break;
            }

            const double to0 = chat_prof ? now_s() : 0.0;
            std::cout << tok_str << std::flush;
            response_text += tok_str;
            if (chat_prof) {
                t_out += now_s() - to0;
            }
            tokens_generated++;
            if (pending_valid) {
                gen_logprobs.push_back(pending_lp);
                pending_valid = false;
            }

            size_t pos = ids.size();
            ids.push_back(static_cast<TokenId>(next_id));
            recent_ids.push_back(next_id);
            if (next_id >= 0 && static_cast<size_t>(next_id) < tok_counts.size()) {
                ++tok_counts[next_id];
            }

            const double tf0 = chat_prof ? now_s() : 0.0;
            core::Tensor next_logits = infer::llama_forward_single_token(model, next_id, pos, &cache);
            if (chat_prof) {
                t_fwd += now_s() - tf0;
            }
            const double ts0 = chat_prof ? now_s() : 0.0;
            if (!gpu_sample(next_logits, 0, next_id)) {
                next_id = sampler.sample(next_logits, recent_ids, repeat_penalty);
            }
            pending_lp = token_logprob(next_logits, 0, next_id);
            pending_valid = true;
            if (chat_prof) {
                t_smp += now_s() - ts0;
            }
        }
        if (chat_prof && tokens_generated > 0) {
            std::cerr << "[chatprof] decode=" << t_dec * 1e3 / tokens_generated
                      << "ms fwd=" << t_fwd * 1e3 / tokens_generated << "ms sampler=" << t_smp * 1e3 / tokens_generated
                      << "ms out=" << t_out * 1e3 / tokens_generated
                      << "ms prefill=" << t_prefill * 1e3 / tokens_generated << "ms/tok\n";
        }
        auto gen_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(gen_end - gen_start).count();
        std::cout << "\n[" << tokens_generated << " tokens generated in " << elapsed << "s ("
                  << (tokens_generated / (elapsed > 0 ? elapsed : 1.0)) << " tok/s)]\n";
#ifdef SERIUS_EXPERIMENTAL_GROUNDING
        if (!gen_logprobs.empty()) {
            const float conf = grounding::confidence_from_logprobs(gen_logprobs);
            std::cout << "[mean token confidence: " << conf << "]\n";
        }
#endif
        if (toxicity_enabled && !response_text.empty()) {
            const float tox = toxicity_classifier.score(response_text);
            std::cout << "[toxicity score: " << tox << "]";
            if (tox >= toxicity_threshold) {
                std::cout << " WARNING: above threshold " << toxicity_threshold;
            }
            std::cout << "\n";
        }
    };

    if (!single_prompt.empty()) {
        std::cout << "User: " << single_prompt << "\nAssistant: ";
        run_turn(single_prompt);
        return 0;
    }

    std::cout << "\n=== Serius Interactive Chat ===\n";
    std::cout << "Type your message and press Enter. Empty line to quit.\n\n";

    while (true) {
        std::cout << "> " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line) || line.empty()) {
            break;
        }
        std::cout << "Assistant: ";
        run_turn(line);
    }
    return 0;
}

}  // namespace cli
}  // namespace serius
