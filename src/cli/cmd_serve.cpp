#include "cli/cmd_serve.h"
#include "cli/args_parser.h"
#include "../common/defaults.h"
#include "../hub/downloader.h"
#include "../server/server.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace serius {
namespace cli {

int cmd_serve(int argc, char **argv)
{
    ArgsParser parser("serius serve", "Start HTTP API server");

    parser.add_flag({"model", "m", "Path to model dir/config/GGUF, or hf://repo_id", true, ""});
    parser.add_flag({"quant", "", "GGUF quant tag for hf:// models (e.g. q4_k_m); default: auto", true, ""});
    parser.add_flag({"type", "", "Model type: 'llama' or 'transformer'", true, "llama"});
    parser.add_flag({"name", "", "Model name reported by /v1/models", true, ""});
    parser.add_flag({"port", "p", "Server port", true, std::to_string(defaults::kHttpPort)});
    parser.add_flag({"host", "", "Server host", true, "0.0.0.0"});
    parser.add_flag({"max-tokens", "", "Default max tokens per request", true, "256"});
    parser.add_flag({"temperature", "", "Default sampling temperature", true, "0.7"});
    parser.add_flag({"top-k", "", "Default top-k sampling (0 = disabled)", true, "40"});
    parser.add_flag({"top-p", "", "Default top-p (nucleus) sampling", true, "0.9"});
    parser.add_flag({"gpu", "", "GPU backend (auto/cpu/cuda)", true, "auto"});
    parser.add_flag({"ui", "", "Web UI directory (default 'ui')", true, ""});

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
    const std::string model_type = parser.get("type", "llama");
    const std::string gpu = parser.get("gpu", "auto");

    if (model_type == "qwen") {
        std::cerr << "Error: model type 'qwen' is not supported by 'serius serve' yet "
                     "(use 'llama' or 'transformer').\n";
        return 1;
    }

    // hf://repo_id → resolve the best GGUF on HuggingFace Hub and download it
    // to models/<repo>/ when missing.
    if (model_path.rfind("hf://", 0) == 0) {
        const std::string repo_id = model_path.substr(5);
        const std::string quant_want = parser.get("quant");

        hub::ModelDownloader dl;
        std::cout << "[Serius] Resolving hf://" << repo_id << " ...\n";
        std::string chosen, local;
        if (!dl.resolve_hf_gguf(repo_id, quant_want, local, chosen)) {
            if (!quant_want.empty()) {
                std::cerr << "[Serius] No GGUF matching --quant=" << quant_want
                          << " in hf://" << repo_id << ".\n";
            } else {
                std::cerr << "[Serius] No .gguf files in hf://" << repo_id
                          << " — use a '-GGUF' repo, or 'serius import' for safetensors.\n";
            }
            return 1;
        }
        model_path = local;
        std::cout << "[Serius] Model: " << model_path << "\n";
    }

    if (!model_path.empty() && !std::filesystem::exists(model_path)) {
        std::cerr << "Error: model path does not exist: " << model_path << "\n";
        return 1;
    }

    server::ServerConfig config;
    config.port = static_cast<uint16_t>(parser.get_int("port", defaults::kHttpPort));
    config.host = parser.get("host", "0.0.0.0");
    config.model_path = model_path;
    config.model_type = model_type;
    const std::string name = parser.get("name");
    if (!name.empty()) {
        config.model_name = name;
    }
    config.max_new_tokens = static_cast<size_t>(parser.get_int("max-tokens", 256));
    config.default_temperature = parser.get_float("temperature", 0.7f);
    config.default_top_k = parser.get_int("top-k", 40);
    config.default_top_p = parser.get_float("top-p", 0.9f);
    const std::string ui_dir = parser.get("ui");
    if (!ui_dir.empty()) {
        config.ui_dir = ui_dir;
    }
    if (gpu == "cpu") {
        config.gpu_mode = server::ServerConfig::GpuMode::kCpu;
        std::cout << "[Serius] CPU inference (--gpu cpu)\n";
    } else if (gpu == "cuda") {
        config.gpu_mode = server::ServerConfig::GpuMode::kCuda;
        std::cout << "[Serius] GPU inference forced (--gpu cuda)\n";
    }

    std::cout << "[Serius] Starting server on " << config.host << ":" << config.port;
    if (!config.model_path.empty()) {
        std::cout << " (model: " << config.model_path << ", type: " << config.model_type << ")";
    } else {
        std::cout << " (no model loaded — API returns fallback responses)";
    }
    std::cout << "\n";

    server::Server server(config);
    return server.run();
}

}  // namespace cli
}  // namespace serius
