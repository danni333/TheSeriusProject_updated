#include "cli/cmd_generate.h"

#include <iostream>
#include <string>
#include <vector>

#include "cli/args_parser.h"
#include "cli/cmd_chat.h"

namespace serius {
namespace cli {

// Non-interactive single-prompt generation. Implemented as a thin wrapper
// over `chat --prompt` so both commands share the exact same inference path.
int cmd_generate(int argc, char **argv)
{
    ArgsParser parser("serius generate", "Generate text from a single prompt (non-interactive)");

    parser.add_flag({"model", "m", "Path to model directory/config, or hf://repo_id", true, "models/smollm2-135m"});
    parser.add_flag({"prompt", "p", "Prompt text", true, ""});
    parser.add_flag({"system", "s", "System prompt", true, "You are a helpful AI assistant."});
    parser.add_flag({"max-tokens", "", "Maximum tokens to generate", true, "128"});
    parser.add_flag({"temperature", "", "Sampling temperature", true, "0.7"});
    parser.add_flag({"top-k", "", "Top-k sampling (0 = disabled)", true, "40"});
    parser.add_flag({"top-p", "", "Top-p (nucleus) sampling", true, "0.9"});
    parser.add_flag({"seed", "", "Random seed for sampling (0 = random)", true, "0"});
    parser.add_flag({"type", "", "Model type: 'llama', 'qwen', or 'transformer'", true, "llama"});
    parser.add_flag({"gpu", "", "GPU backend (auto/cpu/cuda)", true, "auto"});
    parser.add_flag({"gpu-layers", "", "Partial GPU offload: layers on GPU, rest CPU (-1 = auto)", true, "-1"});
    parser.add_flag({"raw", "", "Do not format prompt with ChatML template", false, ""});
    parser.add_flag(
        {"embed-model", "", "Embedding model (GGUF/safetensors dir) enabling semantic scorers (--toxicity)", true, ""});
    parser.add_flag(
        {"toxicity", "", "Score the response for toxicity (0-1); warns above --toxicity-threshold", false, ""});
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

    const std::string prompt = parser.get("prompt");
    if (prompt.empty()) {
        std::cerr << "Error: --prompt is required\n";
        parser.print_help();
        return 1;
    }

    // Rebuild argv for cmd_chat.
    std::vector<std::string> args = {
        "chat",
        "--model",
        parser.get("model"),
        "--prompt",
        prompt,
        "--system",
        parser.get("system"),
        "--max-tokens",
        parser.get("max-tokens"),
        "--temperature",
        parser.get("temperature"),
        "--top-k",
        parser.get("top-k"),
        "--top-p",
        parser.get("top-p"),
        "--seed",
        parser.get("seed"),
        "--type",
        parser.get("type"),
        "--gpu",
        parser.get("gpu"),
        "--gpu-layers",
        parser.get("gpu-layers"),
    };
    const std::string embed_model = parser.get("embed-model");
    if (!embed_model.empty()) {
        args.push_back("--embed-model");
        args.push_back(embed_model);
    }
    if (parser.has("toxicity")) {
        args.push_back("--toxicity");
        args.push_back("--toxicity-threshold");
        args.push_back(parser.get("toxicity-threshold"));
    }
    if (parser.has("raw")) {
        args.push_back("--raw");
    }

    std::vector<char *> argv_out;
    argv_out.reserve(args.size());
    for (auto &s : args) {
        argv_out.push_back(s.data());
    }
    return cmd_chat(static_cast<int>(argv_out.size()), argv_out.data());
}

}  // namespace cli
}  // namespace serius
