#include "cli/cmd_eval.h"
#include "cli/args_parser.h"
#include "cli/llama_loader.h"
#include "eval/perplexity_runner.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace serius {
namespace cli {

int cmd_eval(int argc, char **argv)
{
    ArgsParser parser("serius eval", "Evaluate a model on a task (currently: perplexity)");

    parser.add_flag({"model", "m", "Path to model directory/config, or hf://repo_id", true, "models/smollm2-135m"});
    parser.add_flag({"quant", "", "GGUF quant tag for hf:// models (e.g. q4_k_m); default: auto", true, ""});
    parser.add_flag({"task", "t", "Eval task: 'ppl' (perplexity)", true, "ppl"});
    parser.add_flag({"file", "f", "Corpus file to score", true, ""});
    parser.add_flag({"text", "x", "Inline text to score", true, ""});
    parser.add_flag({"context", "c", "Context tokens per forward pass", true, "512"});
    parser.add_flag({"max-tokens", "n", "Max scored tokens (0 = all)", true, "0"});
    parser.add_flag({"gpu", "g", "GPU backend (auto/cpu/cuda)", true, "auto"});

    if (!parser.parse(argc, argv)) {
        std::cerr << "Error: " << parser.error() << "\n";
        parser.print_help();
        return 1;
    }

    if (parser.has("help")) {
        parser.print_help();
        return 0;
    }

    const std::string task = parser.get("task", "ppl");
    if (task != "ppl") {
        std::cerr << "Error: unknown task '" << task << "' — only 'ppl' is implemented.\n";
        return 1;
    }

    const std::string file_path = parser.get("file");
    const std::string inline_text = parser.get("text");
    if (file_path.empty() == inline_text.empty()) {
        std::cerr << "Error: provide exactly one of --file or --text.\n";
        parser.print_help();
        return 1;
    }

    auto loaded = load_llama_cli_model(parser.get("model"), parser.get("gpu", "auto"),
                                       parser.get("quant"));
    if (!loaded) {
        return 1;
    }

    eval::PerplexityOptions opts;
    opts.context_tokens = static_cast<size_t>(parser.get_int("context", 512));
    opts.max_tokens = static_cast<size_t>(parser.get_int("max-tokens", 0));

    eval::PerplexityResult r;
    if (!file_path.empty()) {
        r = eval::evaluate_perplexity_file(*loaded->model, loaded->tokenizer, file_path, opts);
    } else {
        r = eval::evaluate_perplexity(*loaded->model, loaded->tokenizer, inline_text, opts);
    }

    if (!r.ok) {
        std::cerr << "Eval failed: " << r.error << "\n";
        return 1;
    }

    const double tok_per_s = r.elapsed_seconds > 0.0
                                 ? static_cast<double>(r.scored_tokens) / r.elapsed_seconds
                                 : 0.0;
    std::cout << "\n=== Serius Eval: perplexity ===\n";
    std::cout << "perplexity      : " << r.perplexity << "\n";
    std::cout << "mean_nll        : " << r.mean_nll << "\n";
    std::cout << "scored_tokens   : " << r.scored_tokens << "\n";
    std::cout << "total_tokens    : " << r.total_tokens << "\n";
    std::cout << "chunks          : " << r.chunks << "\n";
    std::cout << "elapsed         : " << r.elapsed_seconds << " s ("
              << tok_per_s << " tok/s)\n";
    return 0;
}

}  // namespace cli
}  // namespace serius
