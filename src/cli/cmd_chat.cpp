#include "cli/cmd_chat.h"
#include "cli/args_parser.h"
#include "infer/sampler.h"
#include "io/config_loader.h"
#include "tokenizer/bpe.h"
#include "transformer/llama_model.h"
#include "transformer/qwen_model.h"
#include "transformer/model.h"
#include "io/hf_llama_loader.h"
#include "infer/llama_inference.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <chrono>
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

    infer::Sampler sampler(temperature, 40, 0.9f, seed);

    if (model_type == "llama") {
        model::LLamaConfig config = io::load_llama_config(config_path);
        if (config.vocab_size == 0) {
            config.vocab_size = tokenizer.vocab_size() > 0 ? tokenizer.vocab_size() : 32000;
        }

        std::cout << "[Serius] Initializing LLaMA model (hidden=" << config.hidden_size
                  << ", layers=" << config.num_hidden_layers
                  << ", heads=" << config.num_attention_heads
                  << ", kv_heads=" << config.num_key_value_heads << ")...\n";

        model::LLaMAModel model(config);

        if (std::filesystem::is_directory(model_dir)) {
            if (io::load_llama_weights_from_dir(model_dir, &model)) {
                std::cout << "[Serius] Successfully loaded .safetensors weights from " << model_dir << "\n";
            } else {
                std::cout << "[Serius] Warning: Could not load weights from " << model_dir << " (using random init)\n";
            }
        }

        bool is_chatml = (tokenizer.get_id("<|im_start|>") != BPETokenizer::UNK_ID);

        auto run_turn = [&](const std::string &user_input) {
            std::string prompt_text = user_input;
            if (is_chatml && !raw_mode) {
                prompt_text = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
            }

            std::vector<TokenId> ids = tokenizer.encode(prompt_text);
            if (!is_chatml && (ids.empty() || ids[0] != BPETokenizer::BOS_ID)) {
                ids.insert(ids.begin(), BPETokenizer::BOS_ID);
            }

            infer::KVCache cache;
            core::Tensor prompt_input({ids.size()});
            for (size_t i = 0; i < ids.size(); ++i) {
                prompt_input[i] = static_cast<float>(ids[i]);
            }

            std::vector<int> recent_ids;
            recent_ids.reserve(ids.size() + max_new_tokens);
            for (TokenId tid : ids) recent_ids.push_back(static_cast<int>(tid));

            auto gen_start = std::chrono::steady_clock::now();
            core::Tensor logits = infer::llama_forward_inference(model, prompt_input, &cache);

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
            for (int gen = 0; gen < max_new_tokens; ++gen) {
                std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
                if (next_id == static_cast<int>(BPETokenizer::EOS_ID) ||
                    tok_str == "<|im_end|>" || tok_str == "<|endoftext|>" ||
                    tok_str == "</s>") {
                    break;
                }

                std::cout << tok_str << std::flush;
                tokens_generated++;

                size_t pos = ids.size();
                ids.push_back(static_cast<TokenId>(next_id));
                recent_ids.push_back(next_id);

                core::Tensor next_logits = infer::llama_forward_single_token(model, next_id, pos, &cache);
                next_id = sampler.sample(next_logits, recent_ids, repeat_penalty);
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
