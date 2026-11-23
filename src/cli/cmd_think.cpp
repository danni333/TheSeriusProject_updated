#include "cli/cmd_think.h"
#include "cli/args_parser.h"
#include "cli/llama_loader.h"
#include "agent/llama_model_adapter.h"
#include "reasoning/agent_model_adapter.h"
#include "reasoning/cot.h"
#include "reasoning/mcts.h"
#include "reasoning/verifier.h"
#include "../hub/search_engine.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace serius {
namespace cli {

namespace {

std::vector<std::string> split_csv(const std::string &s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

}  // namespace

int cmd_think(int argc, char **argv)
{
    ArgsParser parser("serius think", "Chain-of-thought reasoning with a hidden thinking buffer");

    parser.add_flag({"model", "m", "Path to model directory/config, or hf://repo_id", true, "models/smollm2-135m"});
    parser.add_flag({"quant", "", "GGUF quant tag for hf:// models", true, ""});
    parser.add_flag({"prompt", "p", "Question/prompt", true, ""});
    parser.add_flag({"max-thinking-tokens", "", "Max tokens for the hidden thinking phase", true, "2048"});
    parser.add_flag({"show-thinking", "", "Print the thinking buffer (hidden by default)", false, ""});
    parser.add_flag({"gpu", "g", "GPU backend (auto/cpu/cuda)", true, "auto"});
    parser.add_flag({"reference", "", "Reference answer: enables MCTS search scored by EmbeddingVerifier", true, ""});
    parser.add_flag({"embed-model", "", "Embedding model (GGUF/safetensors dir) for the verifier; lexical fallback if unset", true, ""});
    parser.add_flag({"failure-keywords", "", "Comma-separated keywords penalized by the verifier", true, ""});
    parser.add_flag({"mcts-simulations", "", "MCTS simulations per search", true, "8"});

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
        return 1;
    }

    auto loaded = load_llama_cli_model(parser.get("model"), parser.get("gpu", "auto"),
                                       parser.get("quant"));
    if (!loaded) {
        return 1;
    }

    // Real model path: LLaMAModel → agent::Model → reasoning adapters.
    agent::LlamaAgentModel agent_model(*loaded->model, loaded->tokenizer, 42);

    const std::string reference = parser.get("reference");
    if (!reference.empty()) {
        // MCTS test-time compute: search reasoning paths scored against the
        // reference answer by EmbeddingVerifier (semantic when an embed
        // model is provided, lexical overlap otherwise).
        hub::EmbeddingManager embed_mgr;
        common::TextEmbedFn embed_fn;
        const std::string embed_model = parser.get("embed-model");
        if (!embed_model.empty()) {
            embed_mgr.load_model(embed_model);
            if (embed_mgr.is_model_loaded()) {
                embed_fn = embed_mgr.query_embed_fn();
                std::cerr << "[think] embed model loaded: " << embed_model << "\n";
            } else {
                std::cerr << "[think] embed model unavailable, lexical verifier fallback\n";
            }
        }
        reasoning::EmbeddingVerifier verifier(embed_fn, reference,
                                              split_csv(parser.get("failure-keywords")));
        reasoning::AgentMCTSModel mcts_model(agent_model);
        reasoning::MCTSReasoner reasoner(mcts_model, verifier,
                                         parser.get_int("mcts-simulations", 8));
        std::cout << "[Serius] MCTS search (" << parser.get_int("mcts-simulations", 8)
                  << " simulations)...\n";
        std::cout << reasoner.search(prompt) << "\n";
        return 0;
    }

    reasoning::AgentReasoningModel reasoning_model(agent_model);
    reasoning::ChainOfThought cot(reasoning_model,
                                  parser.get_int("max-thinking-tokens", 2048));

    std::cout << "[Serius] Thinking...\n";
    const auto result = cot.generate_with_thinking(prompt);

    if (parser.has("show-thinking") && !result.thinking.empty()) {
        std::cout << "\n--- thinking ---\n" << result.thinking << "\n--- answer ---\n";
    }
    std::cout << result.answer << "\n";
    return 0;
}

}  // namespace cli
}  // namespace serius
