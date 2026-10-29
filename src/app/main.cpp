// TheSeriusProject CLI — serius serve | serius chat | ...
#include "../server/server.h"
#include "../autograd/node.h"
#include "../infer/sampler.h"
#include "../io/config_loader.h"
#include "../tokenizer/bpe.h"
#include "../transformer/llama_model.h"
#include "../transformer/qwen_model.h"
#include "../transformer/model.h"
#include "../io/hf_llama_loader.h"
#include "../infer/llama_inference.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <filesystem>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void print_usage(const char *prog)
{
    std::cerr << "Usage: " << prog << " <command> [options]\n"
              << "  serve [--port PORT]        Start API server (default 8080)\n"
              << "        [--model PATH]       Model directory or config JSON\n"
              << "        [--config PATH]      Alias for --model\n"
              << "        [--type TYPE]        Model type: 'llama', 'qwen', or 'transformer'\n"
              << "  chat  [--config PATH]      Interactive chat with local model\n"
              << "        [--model PATH]       Alias for --config (config JSON)\n"
              << "        [--type TYPE]        Model type: 'transformer', 'llama', or 'qwen'\n"
              << "        [--max-new-tokens N] Limit Transformer/LLaMA output (default 128)\n"
              << "  --help                     Show this help\n";
}

int cmd_serve(int argc, char **argv)
{
    uint16_t port = 8080;
    std::string model_path;
    std::string model_type = "llama";

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::atoi(argv[++i]));
        } else if ((arg == "--model" || arg == "--config") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--type" && i + 1 < argc) {
            model_type = argv[++i];
        }
    }

    serius::server::ServerConfig config;
    config.port = port;
    config.model_path = model_path;
    config.model_type = model_type;

    serius::server::Server server(config);
    return server.run();
}

int cmd_chat(int argc, char **argv)
{
    std::string config_path;
    std::string model_type = "transformer";
    size_t max_new_tokens = 128;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "--model") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--type" && i + 1 < argc) {
            model_type = argv[++i];
        } else if (arg == "--max-new-tokens") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --max-new-tokens.\n";
                return 1;
            }
            const char *value = argv[++i];
            if (*value == '-') {
                std::cerr << "Invalid value for --max-new-tokens: expected a positive integer.\n";
                return 1;
            }
            char *end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (end == value || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
                std::cerr << "Invalid value for --max-new-tokens: expected a positive integer.\n";
                return 1;
            }
            max_new_tokens = static_cast<size_t>(parsed);
        }
    }

    serius::BPETokenizer tokenizer;
    serius::infer::Sampler sampler(0.7f, 40, 0.9f, 42u);

    if (model_type == "llama") {
        serius::model::LLamaConfig config;
        std::string model_dir = ".";
        std::string actual_config_path = config_path;

        if (!config_path.empty()) {
            if (std::filesystem::is_directory(config_path)) {
                model_dir = config_path;
                actual_config_path = config_path + "/config.json";
            } else {
                size_t pos = config_path.find_last_of("/\\");
                if (pos != std::string::npos) {
                    model_dir = config_path.substr(0, pos);
                } else {
                    model_dir = ".";
                }
            }

            config = serius::io::load_llama_config(actual_config_path);
            if (config.vocab_size == 0) {
                config.vocab_size = 32000;
            }

            // Load vocab
            std::string vocab_path = model_dir + "/vocab.json";
            if (!std::filesystem::exists(vocab_path)) {
                vocab_path = model_dir + "/tokenizer.json";
            }
            
            if (std::filesystem::exists(vocab_path)) {
                tokenizer.load(vocab_path);
                std::cout << "Loaded vocab (" << tokenizer.vocab_size() << " tokens) from " << vocab_path << "\n";
            }
        }
        if (tokenizer.vocab_size() < config.vocab_size) {
            std::cerr << "Note: tokenizer vocab size " << tokenizer.vocab_size()
                      << " < model vocab_size " << config.vocab_size << ".\n";
        }
        serius::model::LLaMAModel model(config);

        if (!model_dir.empty()) {
            if (serius::io::load_llama_weights_from_dir(model_dir, &model)) {
                std::cout << "Loaded .safetensors weights from " << model_dir << "\n";
            } else {
                std::cout << "Warning: Failed to load weights from " << model_dir << " (using random init)\n";
            }
        }

        std::cout << "Chat LLaMA (config: hidden_size=" << config.hidden_size << ", layers=" << config.num_hidden_layers
                  << "). Type a line and press Enter. Empty line to quit.\n";

        while (true) {
            std::cout << "> " << std::flush;
            std::string line;
            if (!std::getline(std::cin, line) || line.empty()) {
                break;
            }

            std::vector<serius::TokenId> ids = tokenizer.encode(line);
            // Prepend BOS for LLaMA/Qwen models to ensure proper start-of-sequence
            if (ids.empty() || ids[0] != serius::BPETokenizer::BOS_ID) {
                ids.insert(ids.begin(), serius::BPETokenizer::BOS_ID);
            }

            // Autoregressive generation loop with KV-Caching
            auto gen_start = std::chrono::steady_clock::now();
            size_t tokens_generated = 0;
            
            serius::infer::KVCache cache;
            
            // 1. Initial prompt processing
            serius::core::Tensor prompt_input({ids.size()});
            for (size_t i = 0; i < ids.size(); ++i) prompt_input[i] = static_cast<float>(ids[i]);
            
            // Now llama_forward_inference natively populates the cache in one parallel go.
            cache.clear();
            serius::core::Tensor logits = serius::infer::llama_forward_inference(model, prompt_input, &cache);
            
            int next_id = 0;
            {
                size_t vocab_size = logits.shape().back();
                size_t last_row = logits.shape()[0] - 1;
                serius::core::Tensor last_logits({vocab_size});
                const float *src = logits.data() + (last_row * vocab_size);
                std::copy(src, src + vocab_size, last_logits.data());
                next_id = sampler.sample(last_logits);
            }

            for (size_t gen = 0; gen < max_new_tokens; ++gen) {
                if (next_id == serius::BPETokenizer::EOS_ID) { break; }

                std::string tok_str = tokenizer.decode({static_cast<serius::TokenId>(next_id)});
                std::cout << tok_str << std::flush;

                ++tokens_generated;
                if (gen + 1 == max_new_tokens) {
                    break;
                }
                size_t pos = ids.size();
                ids.push_back(static_cast<serius::TokenId>(next_id));

                // Fast path: use single token inference with KV-cache
                serius::core::Tensor next_logits = 
                    serius::infer::llama_forward_single_token(model, next_id, pos, &cache);
                
                next_id = sampler.sample(next_logits);
            }
            auto gen_end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(gen_end - gen_start).count();
            if (tokens_generated > 0) {
                std::cout << "\n[" << tokens_generated << " tokens in "
                          << elapsed << "s, "
                          << (static_cast<double>(tokens_generated) / elapsed)
                          << " tok/s]\n";
            } else {
                std::cout << "\n";
            }
        }
        return 0;
    } else if (model_type == "qwen") {
        serius::model::QwenConfig config;
        if (!config_path.empty()) {
            config = serius::io::load_qwen_config(config_path);
            if (config.vocab_size == 0) {
                config.vocab_size = 151936;
            }
        }
        if (tokenizer.vocab_size() < config.vocab_size) {
            std::cerr << "Note: tokenizer vocab size " << tokenizer.vocab_size()
                      << " < model vocab_size " << config.vocab_size << ".\n";
        }
        serius::model::QwenModel model(config);

        std::cout << "Chat Qwen (config: hidden_size=" << config.hidden_size << ", layers=" << config.num_hidden_layers
                  << "). Type a line and press Enter. Empty line to quit.\n";

        while (true) {
            std::cout << "> ";
            std::string line;
            if (!std::getline(std::cin, line) || line.empty()) {
                break;
            }

            std::vector<serius::TokenId> ids = tokenizer.encode(line);
            if (ids.empty()) {
                ids.push_back(serius::BPETokenizer::BOS_ID);
            }

            const size_t seq_len = ids.size();
            const size_t max_len = std::min(seq_len, config.max_position_embeddings);
            if (max_len == 0) continue;

            serius::core::Tensor input_tensor({max_len});
            for (size_t i = 0; i < max_len; ++i) {
                input_tensor[i] = static_cast<float>(ids[i]);
            }

            serius::core::autograd::Node input_node(input_tensor, false);
            serius::core::autograd::Node *logits_node = model.forward(&input_node);

            const serius::core::Tensor &logits = logits_node->data;
            size_t vocab_size = logits.shape().back();
            size_t last_row = logits.shape()[0] - 1;

            serius::core::Tensor last_logits({vocab_size});
            const float *src = logits.data() + last_row * vocab_size;
            for (size_t i = 0; i < vocab_size; ++i) {
                last_logits[i] = src[i];
            }

            int next_id = sampler.sample(last_logits);
            std::string out = tokenizer.decode({static_cast<serius::TokenId>(next_id)});
            std::cout << out << std::endl;
        }
        return 0;
    }

    // Default TransformerModel
    serius::model::ModelConfig config;
    if (!config_path.empty()) {
        config = serius::io::load_model_config(config_path);
        config.normalize_moe();
        if (config.vocab_size == 0) {
            config.vocab_size = serius::model::ModelConfig::DEFAULT_VOCAB_SIZE;
        }
    } else {
        config.vocab_size = 320;
        config.d_model = 64;
        config.n_heads = 2;
        config.n_layers = 2;
        config.d_ff = 128;
        config.max_seq_len = 64;
    }

    serius::model::TransformerModel model(config);
    if (tokenizer.vocab_size() < config.vocab_size) {
        std::cerr << "Note: tokenizer vocab size " << tokenizer.vocab_size()
                  << " < model vocab_size " << config.vocab_size << ". Using tokenizer vocab for encode/decode.\n";
    }

    std::cout << "Chat Transformer (config: d_model=" << config.d_model << ", layers=" << config.n_layers;
    if (config.moe_enabled) {
        std::cout << ", moe_enabled=true, experts=" << config.num_experts << ", top_k=" << config.top_k;
    }
    std::cout << "). Type a line and press Enter. Empty line to quit.\n";

    while (true) {
        std::cout << "> ";
        std::string line;
        if (!std::getline(std::cin, line) || line.empty()) {
            break;
        }

        std::vector<serius::TokenId> ids = tokenizer.encode(line);
        if (ids.empty()) {
            ids.push_back(serius::BPETokenizer::BOS_ID);
        }

        const size_t seq_len = ids.size();
        const size_t max_len = std::min(seq_len, config.max_seq_len);
        if (max_len == 0) continue;

        serius::core::Tensor input_tensor({max_len});
        for (size_t i = 0; i < max_len; ++i) {
            input_tensor[i] = static_cast<float>(ids[i]);
        }

        serius::core::autograd::Node input_node(input_tensor, false);
        serius::core::autograd::Node *logits_node = model.forward(&input_node, nullptr);

        const serius::core::Tensor &logits = logits_node->data;
        size_t vocab_size = logits.shape().back();
        size_t last_row = logits.shape()[0] - 1;

        serius::core::Tensor last_logits({vocab_size});
        const float *src = logits.data() + last_row * vocab_size;
        for (size_t i = 0; i < vocab_size; ++i) {
            last_logits[i] = src[i];
        }

        int next_id = sampler.sample(last_logits);

        for (size_t step = 0; step < max_new_tokens; ++step) {
            if (next_id == serius::BPETokenizer::EOS_ID) {
                break;
            }

            std::cout << tokenizer.decode({static_cast<serius::TokenId>(next_id)}) << std::flush;
            if (step + 1 == max_new_tokens) {
                break;
            }

            ids.push_back(static_cast<serius::TokenId>(next_id));
            const size_t cur_len = std::min(ids.size(), config.max_seq_len);

            serius::core::Tensor inp({cur_len});
            for (size_t i = 0; i < cur_len; ++i) {
                inp[i] = static_cast<float>(ids[ids.size() - cur_len + i]);
            }

            serius::core::autograd::Node inp_node(inp, false);
            serius::core::autograd::Node *logits_n = model.forward(&inp_node);

            const serius::core::Tensor &lg = logits_n->data;
            const size_t vs = lg.shape().back();
            const size_t lr = lg.shape()[0] - 1;

            serius::core::Tensor last_lg({vs});
            const float *s = lg.data() + lr * vs;
            for (size_t i = 0; i < vs; ++i) {
                last_lg[i] = s[i];
            }

            next_id = sampler.sample(last_lg);
        }
        std::cout << std::endl;
    }

    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    spdlog::set_level(spdlog::level::info);

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h") {
        print_usage(argv[0]);
        return 0;
    }
    if (cmd == "serve") {
        return cmd_serve(argc, argv);
    }
    if (cmd == "chat") {
        return cmd_chat(argc, argv);
    }

    std::cerr << "Unknown command: " << cmd << "\n";
    print_usage(argv[0]);
    return 1;
}
