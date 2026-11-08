#include "cli/cmd_chat.h"
#include "cli/args_parser.h"
#include "hal/device.h"
#include "infer/sampler.h"
#include "io/config_loader.h"
#include "io/gguf_loader.h"
#include "tokenizer/bpe.h"
#include "transformer/llama_model.h"
#include "transformer/qwen_model.h"
#include "transformer/model.h"
#include "io/hf_llama_loader.h"
#include "infer/llama_inference.h"
#include "hal/fused_kernels.h"
#include "../blas/blas.h"
#ifdef SERIUS_USE_CUDA
#include "../blas/matmul_cuda.h"
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace serius {
namespace cli {

int cmd_chat(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    ArgsParser parser("serius chat", "Interactive chat with the model");

    parser.add_flag({"model", "m", "Path to model directory or config", true, "models/smollm2-135m"});
    parser.add_flag({"type", "", "Model type: 'llama', 'qwen', or 'transformer'", true, "llama"});
    parser.add_flag({"temperature", "", "Sampling temperature", true, "0.7"});
    parser.add_flag({"max-tokens", "", "Maximum tokens to generate", true, "128"});
    parser.add_flag({"seed", "", "Random seed for sampling (0 = random)", true, "0"});
    parser.add_flag({"repeat-penalty", "", "Repetition penalty", true, "1.15"});
    parser.add_flag({"prompt", "p", "Single prompt to evaluate non-interactively", true, ""});
    parser.add_flag({"system", "s", "System prompt", true, "You are a helpful AI assistant named SmolLM, trained by Hugging Face."});
    parser.add_flag({"raw", "", "Do not format prompt with ChatML template", false, ""});

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
    float temperature = parser.get_float("temperature", 0.7f);
    int max_new_tokens = parser.get_int("max-tokens", 128);
    uint64_t seed = parser.get_int("seed", 0);
    float repeat_penalty = parser.get_float("repeat-penalty", 1.15f);
    std::string single_prompt = parser.get("prompt");
    std::string system_prompt = parser.get("system", "You are a helpful AI assistant named SmolLM, trained by Hugging Face.");
    bool raw_mode = parser.has("raw");

    // If seed is 0, use random seed
    if (seed == 0) {
        std::random_device rd;
        seed = rd();
    }

    // Load Tokenizer
    BPETokenizer tokenizer;
    std::string model_dir = model_path;
    std::string config_path = model_path;
    if (std::filesystem::is_directory(model_path)) {
        model_dir = model_path;
        config_path = model_path + "/config.json";
    } else {
        size_t pos = model_path.find_last_of("/\\");
        model_dir = (pos != std::string::npos) ? model_path.substr(0, pos) : ".";
    }

    std::string vocab_path = model_dir + "/tokenizer.json";
    if (!std::filesystem::exists(vocab_path)) {
        vocab_path = model_dir + "/vocab.json";
    }
    if (std::filesystem::exists(vocab_path)) {
        tokenizer.load(vocab_path);
        std::cout << "[Serius] Loaded vocab (" << tokenizer.vocab_size() << " tokens) from " << vocab_path << "\n";
    }

    // If no vocab was loaded from JSON, look for GGUF tokenizer in model_path or model_dir
    if (tokenizer.vocab_size() <= 260) {
        std::string candidate_gguf;
        if (std::filesystem::is_regular_file(model_path) &&
            model_path.size() >= 5 && model_path.compare(model_path.size() - 5, 5, ".gguf") == 0) {
            candidate_gguf = model_path;
        } else if (std::filesystem::is_directory(model_dir)) {
            for (const auto &entry : std::filesystem::directory_iterator(model_dir)) {
                std::string p = entry.path().string();
                if (p.size() >= 5 && p.compare(p.size() - 5, 5, ".gguf") == 0) {
                    candidate_gguf = p;
                    break;
                }
            }
        }

        if (!candidate_gguf.empty()) {
            auto tok_data = io::load_gguf_tokenizer(candidate_gguf);
            if (tok_data.valid) {
                tokenizer.load_from_gguf(tok_data);
                std::cout << "[Serius] Loaded vocab (" << tokenizer.vocab_size() << " tokens) from GGUF " << candidate_gguf << "\n";
            }
        }
    }

    infer::Sampler sampler(temperature, 40, 0.9f, seed);

    if (model_type == "llama") {
        model::LLamaConfig config;
        // config_path may point at a .gguf file itself — only treat it as a
        // JSON config when it actually ends in .json.
        bool has_config_json = config_path.size() >= 5 &&
            config_path.compare(config_path.size() - 5, 5, ".json") == 0 &&
            std::filesystem::exists(config_path);
        if (has_config_json) {
            config = io::load_llama_config(config_path);
        }

        // If config.json is missing (common with GGUF-only distributions),
        // extract model dimensions directly from the GGUF metadata.
        if (!has_config_json) {
            std::string candidate_gguf;
            if (std::filesystem::is_regular_file(model_path) &&
                model_path.size() >= 5 && model_path.compare(model_path.size() - 5, 5, ".gguf") == 0) {
                candidate_gguf = model_path;
            } else if (std::filesystem::is_directory(model_dir)) {
                for (const auto &entry : std::filesystem::directory_iterator(model_dir)) {
                    std::string p = entry.path().string();
                    if (p.size() >= 5 && p.compare(p.size() - 5, 5, ".gguf") == 0) {
                        candidate_gguf = p;
                        break;
                    }
                }
            }

            if (!candidate_gguf.empty()) {
                auto mc = io::gguf_to_model_config(candidate_gguf);
                if (mc.vocab_size > 0) {
                    config.vocab_size        = mc.vocab_size;
                    config.hidden_size       = mc.d_model;
                    config.num_hidden_layers = mc.n_layers;
                    config.num_attention_heads = mc.n_heads;
                    config.num_key_value_heads = mc.n_kv_heads;
                    config.intermediate_size   = mc.d_ff;
                    config.max_position_embeddings = mc.max_seq_len;
                    config.rope_theta          = mc.rope_theta;
                    config.rms_norm_eps        = mc.rms_norm_eps;
                    // GGUF has no reliable vocab_size key for all archs; the
                    // tokenizer (loaded from the same file) knows the truth.
                    if (tokenizer.vocab_size() > 0) {
                        config.vocab_size = tokenizer.vocab_size();
                    }
                    std::cout << "[Serius] Loaded model config from GGUF metadata (" << candidate_gguf << ")\n";
                }
            }
        }

        if (config.vocab_size == 0) {
            config.vocab_size = tokenizer.vocab_size() > 0 ? tokenizer.vocab_size() : 32000;
        }

        std::cout << "[Serius] Initializing LLaMA model (hidden=" << config.hidden_size
                  << ", layers=" << config.num_hidden_layers
                  << ", heads=" << config.num_attention_heads
                  << ", kv_heads=" << config.num_key_value_heads << ")...\n";

        model::LLaMAModel model(config);

        if (std::filesystem::is_regular_file(model_path) &&
            model_path.size() >= 5 && model_path.compare(model_path.size() - 5, 5, ".gguf") == 0) {
            if (io::load_llama_weights_from_gguf_file(model_path, &model)) {
                std::cout << "[Serius] Successfully loaded weights from GGUF file " << model_path << "\n";
            } else {
                std::cout << "[Serius] Warning: Could not load weights from " << model_path << " (using random init)\n";
            }
        } else if (std::filesystem::is_directory(model_dir)) {
            if (io::load_llama_weights_from_dir(model_dir, &model)) {
                std::cout << "[Serius] Successfully loaded weights from " << model_dir << "\n";
            } else {
                std::cout << "[Serius] Warning: Could not load weights from " << model_dir << " (using random init)\n";
            }
        }

#ifdef SERIUS_USE_CUDA
        if (std::getenv("SERIUS_NO_GPU") == nullptr) {
            // Keep an FP32 transposed lm_head [vocab, hidden] on the GPU:
            // the dedicated logits GEMV kernel reads it coalesced, and FP32
            // preserves the fine-grained ranking over the 150k+ vocabulary.
            // The FP32 transposed lm_head (~933 MB VRAM) is only read by the
            // non-quantized logits fallback; skip it when a packed lm_head
            // exists. The FP16 copy is still needed by the prefill logits GEMV.
            const bool need_fp32_lm_head = !model.lm_head_q().valid();
            if (model.lm_head() != nullptr) {
                core::Tensor w_t = model.lm_head()->data.transpose(0, 1).contiguous();
                if (need_fp32_lm_head) {
                    model.lm_head_fp32() = w_t;
                    model.lm_head_fp32().set_device(hal::DeviceType::CUDA);
                }
                // FP16 copy for the cuBLAS mixed-precision logits GEMV (half the
                // memory traffic of the FP32 path; FP32 accumulation preserved).
                model.lm_head_fp16_t() = w_t.cast(hal::DType::Float16);
                model.lm_head_fp16_t().set_device(hal::DeviceType::CUDA);
            }
            auto move_param = [](core::autograd::Node *p) {
                if (p == nullptr) { return; }
                p->data = p->data.cast(hal::DType::Float16);
                p->data.set_device(hal::DeviceType::CUDA);
            };
            move_param(model.embed_tokens_raw());
            for (auto &blk : model.layers()) {
                move_param(blk->input_layernorm_weight.get());
                move_param(blk->q_proj.get());
                move_param(blk->k_proj.get());
                move_param(blk->v_proj.get());
                move_param(blk->q_bias.get());
                move_param(blk->k_bias.get());
                move_param(blk->v_bias.get());
                move_param(blk->o_proj.get());
                move_param(blk->post_attention_layernorm_weight.get());
                move_param(blk->gate_proj.get());
                move_param(blk->up_proj.get());
                move_param(blk->down_proj.get());
            }
            move_param(model.norm_weight());
            // Upload packed GGUF quantized weights (Q4_0/Q4_1/Q8_0) for the
            // CUDA decode GEMV kernels; no-op for non-quantized models.
            if (std::getenv("SERIUS_NO_QUANT") == nullptr) {
                hal::IBackend *be = hal::get_gpu_backend();
                auto upload_qw = [be](quant::QuantWeight &qw) {
                    if (!qw.valid() || qw.gpu_data != nullptr) { return; }
                    // Repack into the aligned layout the vectorized kernel reads.
                    quant::repack_quant_for_gpu(qw);
                    const std::vector<uint8_t> &src = qw.dev_stride ? qw.dev_bytes : qw.bytes;
                    qw.gpu_data = be->malloc(src.size());
                    be->memcpy_h2d(qw.gpu_data, src.data(), src.size());
                };
                for (auto &blk : model.layers()) {
                    upload_qw(blk->qkv_w);
                    upload_qw(blk->gate_up_w);
                    upload_qw(blk->o_w);
                    upload_qw(blk->down_w);
                }
                upload_qw(model.lm_head_q());
            }
            std::cout << "[Serius] Model moved to GPU (FP16)\n";

            // cuBLAS warmup: the first GemmEx/GemmStridedBatchedEx per kernel
            // variant lazily loads driver modules (cuLibraryLoadData, ~300 ms
            // total) — pay it here during weight upload instead of inside the
            // first prompt's prefill. Shapes mirror the real prefill GEMMs.
            {
                hal::IBackend *gbe = hal::get_gpu_backend();
                const size_t hidden = model.config().hidden_size;
                const size_t inter = model.config().intermediate_size;
                const size_t heads = model.config().num_attention_heads;
                const size_t head_dim = hidden / heads;
                const size_t wm = 256;  // representative M (seq_len)
                const size_t max_k = (std::max)(hidden, inter);
                void *buf = gbe->malloc(wm * max_k * 2 + 4 * 1024 * 1024);
                auto f16_view = [&](size_t m, size_t k) {
                    return core::Tensor::gpu_view(buf, {m, k}, hal::DType::Float16);
                };
                const auto &l0 = model.layers()[0];
                struct WP { core::autograd::Node *w; } wps[] = {
                    {l0->q_proj.get()}, {l0->k_proj.get()}, {l0->v_proj.get()},
                    {l0->o_proj.get()}, {l0->gate_proj.get()}, {l0->up_proj.get()},
                    {l0->down_proj.get()},
                };
                for (auto &wp : wps) {
                    if (wp.w == nullptr) continue;
                    const auto &ws = wp.w->data.shape();
                    const size_t K = ws[0], N = ws.size() > 1 ? ws[1] : ws[0];
                    core::Tensor A = f16_view(wm, K);
                    core::Tensor C = core::Tensor::gpu_view(buf, {wm, N}, hal::DType::Float16);
                    core::blas::matmul(A, wp.w->data, C);
                }
                // Batched attention shapes: {heads, seq, head_dim} variants.
                {
                    core::Tensor A = core::Tensor::gpu_view(buf, {heads, wm, head_dim}, hal::DType::Float16);
                    core::Tensor B = core::Tensor::gpu_view(buf, {heads, head_dim, wm}, hal::DType::Float16);
                    core::Tensor C = core::Tensor::gpu_view(buf, {heads, wm, wm}, hal::DType::Float16);
                    core::blas::batched_matmul(A, B, C);
                    core::Tensor D = core::Tensor::gpu_view(buf, {heads, wm, head_dim}, hal::DType::Float16);
                    core::blas::batched_matmul(C, A, D);
                }
                // Logits GEMV (M=1 path used by llama_forward_inference_last).
                if (model.lm_head_fp16_t().data_gpu() != nullptr) {
                    core::Tensor x = f16_view(1, hidden);
                    core::Tensor out = core::Tensor::gpu_view(buf, {1, model.config().vocab_size}, hal::DType::Float32);
                    core::blas::logits_gemv_mixed_cublas(x.raw_data_gpu(), model.lm_head_fp16_t().raw_data_gpu(),
                                                         out.raw_data_gpu(), 1, model.config().vocab_size, hidden);
                }
                cudaDeviceSynchronize();
                gbe->free(buf);
            }

            // Real-path prefill warmup: pays the one-time costs the shape
            // warmup above can't — first-touch page faults on the prefill
            // scratch arena, custom kernel module loads, and cuBLASLt
            // heuristic selection for the actual projection shapes.
            {
                infer::KVCache warm_cache;
                const size_t wn = 128;
                core::Tensor warm_in({wn});
                for (size_t i = 0; i < wn; ++i) {
                    warm_in[i] = static_cast<float>(1000 + (i % 1000));
                }
                (void)infer::llama_forward_inference_last(model, warm_in, &warm_cache);
            }
        }
#endif

        bool is_chatml = (tokenizer.get_id("<|im_start|>") != BPETokenizer::UNK_ID);

        // Reuse one KVCache across turns: the GPU arena + per-layer views
        // persist, so the CUDA decode graph (bound to those raw pointers)
        // stays valid instead of being recaptured every turn.
        infer::KVCache cache;

#ifdef SERIUS_USE_CUDA
        if (std::getenv("SERIUS_NO_GPU") == nullptr) {
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
                prompt_text = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
            }

            std::vector<TokenId> ids = tokenizer.encode(prompt_text);
            {
                size_t _n = 0;
                if (getenv_s(&_n, nullptr, 0, "SERIUS_PROF") == 0 && _n > 0) {
                    std::fprintf(stderr, "[chatprof] prompt_tokens=%zu\n", ids.size());
                }
            }
            if (!is_chatml && (ids.empty() || ids[0] != BPETokenizer::BOS_ID)) {
                ids.insert(ids.begin(), BPETokenizer::BOS_ID);
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
                if (tid >= 0 && static_cast<size_t>(tid) < tok_counts.size()) { ++tok_counts[tid]; }
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
                if (gp == nullptr) { return false; }
                const size_t vs = lg.shape().back();
                const float *lgp = static_cast<const float *>(gp) + row0 * vs;
                thread_local std::vector<int> cidx;
                thread_local std::vector<float> cval;
                cidx.assign(16384, 0);
                cval.assign(16384, 0.0f);
                int cnt = hal::gpu_topk_raw_candidates(lgp, static_cast<int>(vs), 16384, 2048,
                                                       cidx.data(), cval.data());
                if (cnt <= 0) {
                    // Cooperative path unavailable/failed — iterative fallback.
                    cnt = hal::topk_logits_f32(lgp, static_cast<int>(vs), 384, 16384,
                                               cidx.data(), cval.data());
                    if (cnt <= 0) { return false; }
                }
                if (cnt <= 0) { return false; }
                cidx.resize(static_cast<size_t>(cnt));
                cval.resize(static_cast<size_t>(cnt));
                const int t = sampler.sample_from_candidates_dense(
                    cidx, cval, tok_counts.data(), tok_counts.size(), repeat_penalty);
                if (t < 0) { return false; }
                out_id = t;
                return true;
#else
                return false;
#endif
            };

            int next_id = 0;
            {
                size_t vocab_size = logits.shape().back();
                size_t last_row = logits.shape()[0] - 1;
                core::Tensor last_logits({vocab_size});
                const float *src = logits.data() + (last_row * vocab_size);
                std::copy(src, src + vocab_size, last_logits.data());
                next_id = sampler.sample(last_logits, recent_ids, repeat_penalty);
            }

            size_t tokens_generated = 0;
            const bool chat_prof = [] {
                size_t n = 0;
                return getenv_s(&n, nullptr, 0, "SERIUS_PROF") == 0 && n > 0;
            }();
            double t_dec = 0, t_fwd = 0, t_smp = 0, t_out = 0;
            auto now_s = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
            for (int gen = 0; gen < max_new_tokens; ++gen) {
                const double td0 = chat_prof ? now_s() : 0.0;
                std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
                if (chat_prof) { t_dec += now_s() - td0; }
                // EOS_ID==2 is the GPT-2/LLaMA convention; ChatML models
                // (Qwen, SmolLM) end on <|im_end|>. In the Qwen vocab token 2
                // decodes to "</s>" but is a normal token there, so the legacy
                // stop strings must only apply to non-ChatML models.
                if ((!is_chatml && (next_id == static_cast<int>(BPETokenizer::EOS_ID) ||
                                    tok_str == "<|endoftext|>" ||
                                    tok_str == "</s>")) ||
                    tok_str == "<|im_end|>") {
                    break;
                }

                const double to0 = chat_prof ? now_s() : 0.0;
                std::cout << tok_str << std::flush;
                if (chat_prof) { t_out += now_s() - to0; }
                tokens_generated++;

                size_t pos = ids.size();
                ids.push_back(static_cast<TokenId>(next_id));
                recent_ids.push_back(next_id);
                if (next_id >= 0 && static_cast<size_t>(next_id) < tok_counts.size()) { ++tok_counts[next_id]; }

                const double tf0 = chat_prof ? now_s() : 0.0;
                core::Tensor next_logits = infer::llama_forward_single_token(model, next_id, pos, &cache);
                if (chat_prof) { t_fwd += now_s() - tf0; }
                const double ts0 = chat_prof ? now_s() : 0.0;
                if (!gpu_sample(next_logits, 0, next_id)) {
                    next_id = sampler.sample(next_logits, recent_ids, repeat_penalty);
                }
                if (chat_prof) { t_smp += now_s() - ts0; }
            }
            if (chat_prof && tokens_generated > 0) {
                std::cerr << "[chatprof] decode=" << t_dec * 1e3 / tokens_generated
                          << "ms fwd=" << t_fwd * 1e3 / tokens_generated
                          << "ms sampler=" << t_smp * 1e3 / tokens_generated
                          << "ms out=" << t_out * 1e3 / tokens_generated
                          << "ms prefill=" << t_prefill * 1e3 / tokens_generated << "ms/tok\n";
            }
            auto gen_end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(gen_end - gen_start).count();
            std::cout << "\n[" << tokens_generated << " tokens generated in " << elapsed
                      << "s (" << (tokens_generated / (elapsed > 0 ? elapsed : 1.0)) << " tok/s)]\n";
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

    std::cerr << "Model type '" << model_type << "' not yet fully wired in CLI chat.\n";
    return 1;
}

}  // namespace cli
}  // namespace serius
