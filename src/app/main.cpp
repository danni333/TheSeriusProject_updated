// TheSeriusProject CLI — serius serve | serius chat | ...
#include "../server/server.h"
#include "autograd/node.h"
#include "infer/sampler.h"
#include "io/config_loader.h"
#include "tokenizer/bpe.h"
#include "transformer/model.h"
#include <spdlog/spdlog.h>

#include <cstdlib>
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
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "--model") && i + 1 < argc) {
            config_path = argv[++i];
            break;
        }
    }

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
    serius::BPETokenizer tokenizer;
    if (tokenizer.vocab_size() < config.vocab_size) {
        std::cerr << "Note: tokenizer vocab size " << tokenizer.vocab_size()
                  << " < model vocab_size " << config.vocab_size << ". Using tokenizer vocab for encode/decode.\n";
    }

    serius::infer::Sampler sampler(0.7f, 40, 0.9f, 42u);

    std::cout << "Chat (config: d_model=" << config.d_model << ", layers=" << config.n_layers
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
        if (max_len == 0) {
            continue;
        }

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
