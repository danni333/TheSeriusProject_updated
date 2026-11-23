#include "cli/llama_loader.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "io/config_loader.h"
#include "io/gguf_loader.h"
#include "io/hf_llama_loader.h"
#include "infer/llama_inference.h"
#include "tokenizer/bpe.h"
#include "transformer/llama_model.h"
#include "../hub/downloader.h"

namespace serius {
namespace cli {

std::unique_ptr<LlamaCliModel> load_llama_cli_model(const std::string &model_path_in,
                                                    const std::string &gpu_mode,
                                                    const std::string &quant_tag,
                                                    int gpu_layers)
{
    std::string model_path = model_path_in;

    // hf://repo_id → resolve the best GGUF on HuggingFace Hub, download it to
    // models/<repo>/ if missing, then load the local file like any other path.
    if (model_path.rfind("hf://", 0) == 0) {
        const std::string repo_id = model_path.substr(5);

        hub::ModelDownloader dl;
        std::cout << "[Serius] Resolving hf://" << repo_id << " ...\n";
        std::string chosen, local;
        if (!dl.resolve_hf_gguf(repo_id, quant_tag, local, chosen)) {
            if (!quant_tag.empty()) {
                std::cerr << "[Serius] No GGUF matching --quant=" << quant_tag
                          << " in hf://" << repo_id << ".\n";
            } else {
                std::cerr << "[Serius] No .gguf files in hf://" << repo_id
                          << " — use a '-GGUF' repo, or 'serius import' for safetensors.\n";
            }
            return nullptr;
        }

        model_path = local;
        std::cout << "[Serius] Model: " << model_path << "\n";
    }

    auto out = std::make_unique<LlamaCliModel>();
    out->resolved_path = model_path;
    BPETokenizer &tokenizer = out->tokenizer;

    // Load Tokenizer
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

    auto model = std::make_unique<model::LLaMAModel>(config);

    if (std::filesystem::is_regular_file(model_path) &&
        model_path.size() >= 5 && model_path.compare(model_path.size() - 5, 5, ".gguf") == 0) {
        if (io::load_llama_weights_from_gguf_file(model_path, model.get())) {
            out->weights_loaded = true;
            std::cout << "[Serius] Successfully loaded weights from GGUF file " << model_path << "\n";
        } else {
            std::cout << "[Serius] Warning: Could not load weights from " << model_path << " (using random init)\n";
        }
    } else if (std::filesystem::is_directory(model_dir)) {
        if (io::load_llama_weights_from_dir(model_dir, model.get())) {
            out->weights_loaded = true;
            std::cout << "[Serius] Successfully loaded weights from " << model_dir << "\n";
        } else {
            std::cout << "[Serius] Warning: Could not load weights from " << model_dir << " (using random init)\n";
        }
    }

    // --gpu-layers N: partial offload of the first N blocks (llama.cpp
    // --n-gpu-layers style). N >= layer count falls through to full offload.
    if (gpu_layers > 0 && gpu_mode != "cpu") {
        const size_t nl = static_cast<size_t>(gpu_layers);
        if (nl >= model->layers().size()) {
            std::cout << "[Serius] --gpu-layers " << gpu_layers
                      << " >= layer count — using full GPU offload decision\n";
        } else if (infer::offload_llama_model_partial(*model, nl)) {
            out->model = std::move(model);
            return out;
        } else {
            std::cerr << "[Serius] Partial GPU offload unavailable (CUDA off or bad range)"
                      << " — staying on CPU\n";
        }
    }

    const auto gpu_mode_enum = (gpu_mode == "cpu")   ? infer::LlamaGpuMode::kCpu
                             : (gpu_mode == "cuda") ? infer::LlamaGpuMode::kCuda
                                                    : infer::LlamaGpuMode::kAuto;
    const auto gpu_dec = infer::resolve_llama_gpu(*model, gpu_mode_enum);
    if (gpu_dec.use_gpu) {
        if (gpu_dec.fit == infer::LlamaGpuFit::kInsufficientVram) {
            std::cerr << "[Serius] Warning: GPU forced but estimate needs ~"
                      << gpu_dec.need_bytes / 1073741824.0 << " GiB vs "
                      << gpu_dec.free_bytes / 1073741824.0 << " GiB free\n";
        }
        try {
            if (infer::offload_llama_model_to_gpu(*model)) {
                std::cout << "[Serius] Model ~" << gpu_dec.est_params / 1e9
                          << "B params moved to GPU (~" << gpu_dec.need_bytes / 1073741824.0
                          << " GiB, " << gpu_dec.reason << ")\n";
            }
        } catch (const std::exception &e) {
            std::cerr << "[Serius] GPU offload failed: " << e.what() << " — CPU fallback\n";
        }
    } else if (gpu_dec.fit == infer::LlamaGpuFit::kInsufficientVram) {
        std::cerr << "[Serius] Model ~" << gpu_dec.est_params / 1e9
                  << "B params needs ~" << gpu_dec.need_bytes / 1073741824.0
                  << " GiB VRAM but only " << gpu_dec.free_bytes / 1073741824.0
                  << " GiB free — staying on CPU (" << gpu_dec.reason << ")\n";
    } else if (gpu_mode == "cpu") {
        std::cout << "[Serius] CPU inference (--gpu cpu)\n";
    }

    out->model = std::move(model);
    return out;
}

}  // namespace cli
}  // namespace serius
