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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char *prog)
{
    std::cerr << "Usage: " << prog << " <command> [options]\n"
              << "  serve [--port PORT]       Start API server (default 8080)\n"
              << "  chat [--config PATH]       Interactive chat with local model\n"
              << "  chat [--model PATH]        Alias for --config (config JSON)\n"
              << "       [--type TYPE]         Model type: 'transformer', 'llama', or 'qwen'\n"
              << "  --help                     Show this help\n";
}

int cmd_serve(int argc, char **argv)
{
    uint16_t port = 8080;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::atoi(argv[++i]));
        }
    }
    serius::server::Server server(port);
    return server.run();
}

int cmd_chat(int argc, char **argv)
{
    std::string config_path;
    std::string model_type = "transformer";
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "--model") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--type" && i + 1 < argc) {
            model_type = argv[++i];
        }
    }

    serius::BPETokenizer tokenizer;
    serius::infer::Sampler sampler(0.7f, 40, 0.9f, 42u);

    if (model_type == "llama") {
        serius::model::LLamaConfig config;
        std::string model_dir = ".";
        if (!config_path.empty()) {
            config = serius::io::load_llama_config(config_path);
            if (config.vocab_size == 0) {
                config.vocab_size = 32000;
            }
            size_t pos = config_path.find_last_of("/\\");
            if (pos != std::string::npos) {
                model_dir = config_path.substr(0, pos);
            }
            // Load vocab FIRST so the size check is accurate
            std::string vocab_path = model_dir + "/vocab.json";
            std::ifstream vocab_file(vocab_path);
            if (vocab_file.good()) {
                vocab_file.close();
                tokenizer.load(vocab_path);
                std::cout << "Loaded vocab (" << tokenizer.vocab_size() << " tokens) from " << vocab_path << "\n";
            }
        }
        if (tokenizer.vocab_size() < config.vocab_size) {
            std::cerr << "Note: tokenizer vocab size " << tokenizer.vocab_size()
                      << " < model vocab_size " << config.vocab_size << ".\n";
        }
        serius::model::LLaMAModel model(config);

        if (!config_path.empty()) {
            if (serius::io::load_llama_weights_from_dir(model_dir, &model)) {
                std::cout << "Loaded .safetensors weights from " << model_dir << "\n";
            } else {
                std::cout << "Warning: Failed to load weights from " << model_dir << " (using random init)\n";
            }
        }

        std::cout << "Chat LLaMA (config: hidden_size=" << config.hidden_size << ", layers=" << config.num_hidden_layers
                  << "). Type a line and press Enter. Empty line to quit.\n";

        constexpr size_t MAX_GEN_TOKENS = 128;

        while (true) {
            std::cout << "> " << std::flush;
            std::string line;
            if (!std::getline(std::cin, line) || line.empty()) {
                break;
            }

            std::vector<serius::TokenId> ids = tokenizer.encode(line);
            if (ids.empty()) {
                ids.push_back(serius::BPETokenizer::BOS_ID);
            }

            // Autoregressive generation loop (inference-only, no autograd)
            auto gen_start = std::chrono::steady_clock::now();
            size_t tokens_generated = 0;

            for (size_t gen = 0; gen < MAX_GEN_TOKENS; ++gen) {
                size_t cur_len = ids.size();
                size_t ctx_len = std::min(cur_len, config.max_position_embeddings);
                if (ctx_len == 0) { break; }

                // Use only the last ctx_len tokens
                size_t start = cur_len - ctx_len;
                serius::core::Tensor input_tensor({ctx_len});
                for (size_t i = 0; i < ctx_len; ++i) {
                    input_tensor[i] = static_cast<float>(ids[start + i]);
                }

                // Use inference-only forward pass (no Node/autograd overhead)
                serius::core::Tensor logits =
                    serius::infer::llama_forward_inference(model, input_tensor);

                size_t vocab_size = logits.shape().back();
                size_t last_row = logits.shape()[0] - 1;

                serius::core::Tensor last_logits({vocab_size});
                const float *src = logits.data() + (last_row * vocab_size);
                for (size_t i = 0; i < vocab_size; ++i) {
                    last_logits[i] = src[i];
                }

                int next_id = sampler.sample(last_logits);

                // Stop on EOS
                if (next_id == serius::BPETokenizer::EOS_ID) { break; }

                std::string tok_str = tokenizer.decode({static_cast<serius::TokenId>(next_id)});
                std::cout << tok_str << std::flush;

                ids.push_back(static_cast<serius::TokenId>(next_id));
                ++tokens_generated;
            }
            auto gen_end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(gen_end - gen_start).count();
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

    std::cout << "Chat Transformer (config: d_model=" << config.d_model << ", layers=" << config.n_layers
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
        std::string out = tokenizer.decode({static_cast<serius::TokenId>(next_id)});
        std::cout << out << std::endl;
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
