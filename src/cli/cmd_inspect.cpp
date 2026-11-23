#include "cli/cmd_inspect.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "cli/args_parser.h"
#include "cli/llama_loader.h"
#include "interpret/inference_capture.h"
#include "interpret/logit_lens.h"
#include "tensor/tensor.h"

using serius::core::Tensor;

namespace serius {
namespace cli {

namespace {

/// Mean attention distribution across heads for the LAST captured query
/// row of a layer. Returns probs[k] for key position k, or empty.
std::vector<float> mean_last_attention(const interpret::InferenceCapture &cap, int layer)
{
    const size_t heads = cap.num_attention_heads(layer);
    if (heads == 0) {
        return {};
    }

    std::vector<float> acc;
    size_t npos = 0;
    size_t rows_used = 0;
    for (size_t h = 0; h < heads; ++h) {
        const auto &rows = cap.attention_rows(layer, static_cast<int>(h));
        if (rows.empty()) {
            continue;
        }
        const Tensor &last = rows.back();  // {npos}
        if (last.size() == 0) {
            continue;
        }
        if (acc.empty()) {
            npos = last.shape()[0];
            acc.assign(npos, 0.0f);
        }
        const size_t n = std::min(npos, last.shape()[0]);
        for (size_t k = 0; k < n; ++k) {
            acc[k] += last[k];
        }
        ++rows_used;
    }
    if (rows_used == 0) {
        return {};
    }
    for (float &v : acc) {
        v /= static_cast<float>(rows_used);
    }
    return acc;
}

/// Shannon entropy (nats) of a probability vector.
double entropy_nats(const std::vector<float> &p)
{
    double h = 0.0;
    for (float v : p) {
        if (v > 1e-12f) {
            h -= static_cast<double>(v) * std::log(static_cast<double>(v));
        }
    }
    return h;
}

std::string tok_preview(const std::string &s)
{
    std::string out;
    for (char c : s) {
        if (c == '\n' || c == '\r' || c == '\t') {
            out += ' ';
        } else if (static_cast<unsigned char>(c) < 0x20) {
            out += '?';
        } else {
            out += c;
        }
    }
    if (out.size() > 24) {
        out.resize(24);
        out += "...";
    }
    return out;
}

}  // namespace

int cmd_inspect(int argc, char **argv)
{
    ArgsParser parser("serius inspect", "Inspect per-layer predictions (logit lens) and attention of a prompt");

    parser.add_flag({"model", "m", "Path to model directory/config, or hf://repo_id", true, "models/smollm2-135m"});
    parser.add_flag({"quant", "", "GGUF quant tag for hf:// models (e.g. q4_k_m); default: auto", true, ""});
    parser.add_flag({"prompt", "p", "Prompt text to inspect", true, ""});
    parser.add_flag({"topk", "k", "Top-k predictions per layer (logit lens)", true, "3"});
    parser.add_flag({"attn-top", "a", "Top attended positions per layer", true, "3"});
    parser.add_flag({"gpu", "g", "GPU backend (auto/cpu/cuda) — hooks are CPU-only", true, "cpu"});

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
        std::cerr << "Error: --prompt must not be empty.\n";
        return 1;
    }

    auto loaded = load_llama_cli_model(parser.get("model"), parser.get("gpu", "cpu"), parser.get("quant"));
    if (!loaded) {
        return 1;
    }
    // Note: activation hooks are only emitted by the CPU inference path.
    // --gpu defaults to cpu; cuda/auto may offload and yield no captures.

    const int topk = parser.get_int("topk", 3);
    const int attn_top = parser.get_int("attn-top", 3);

    // Encode prompt -> token id tensor {seq_len} (float, as inference expects)
    const auto ids = loaded->tokenizer.encode(prompt);
    if (ids.empty()) {
        std::cerr << "Error: prompt encoded to zero tokens.\n";
        return 1;
    }
    Tensor token_ids({ids.size()});
    for (size_t i = 0; i < ids.size(); ++i) {
        token_ids[static_cast<size_t>(i)] = static_cast<float>(ids[i]);
    }

    interpret::InferenceCapture capture;
    capture.run(*loaded->model, token_ids);

    const size_t num_layers = capture.num_layers();
    if (num_layers == 0) {
        std::cerr << "Error: no layers captured (is the model using the CPU path?).\n";
        return 1;
    }

    // Logit lens setup: vocab strings + {vocab, hidden} head.
    interpret::LogitLens lens;
    {
        const size_t vs = loaded->tokenizer.vocab_size();
        std::vector<std::string> vocab;
        vocab.reserve(vs);
        for (size_t i = 0; i < vs; ++i) {
            vocab.push_back(loaded->tokenizer.get_token(static_cast<int>(i)));
        }
        lens.set_vocab(vocab);
    }
    const Tensor lm_head = interpret::logit_lens_head(*loaded->model);
    const bool has_lens = lm_head.size() > 0;
    if (!has_lens) {
        std::cerr << "Warning: model has no LM head; logit-lens columns skipped.\n";
    }

    std::cout << "\n=== Serius Inspect ===\n";
    std::cout << "model   : " << parser.get("model") << "\n";
    std::cout << "prompt  : \"" << tok_preview(prompt) << "\"\n";
    std::cout << "tokens  : " << ids.size() << " | layers: " << num_layers << "\n\n";

    for (size_t l = 0; l < num_layers; ++l) {
        std::cout << "layer " << l;

        if (has_lens) {
            const Tensor hidden = capture.hidden_last_position(static_cast<int>(l));
            if (hidden.size() > 0) {
                const auto preds = lens.compute(hidden, lm_head, topk);
                std::cout << "  lens:";
                for (const auto &p : preds) {
                    std::cout << " \"" << tok_preview(p.token) << "\"(" << p.logit << ")";
                }
            }
        }

        const auto attn = mean_last_attention(capture, static_cast<int>(l));
        if (!attn.empty()) {
            const double ent = entropy_nats(attn);
            // top attended positions
            std::vector<size_t> order(attn.size());
            for (size_t i = 0; i < order.size(); ++i) {
                order[i] = i;
            }
            const size_t k = std::min(static_cast<size_t>(attn_top), order.size());
            std::partial_sort(order.begin(), order.begin() + static_cast<long>(k), order.end(),
                              [&](size_t a, size_t b) { return attn[a] > attn[b]; });
            std::cout << "  | attn H=" << ent << " top:";
            for (size_t i = 0; i < k; ++i) {
                const size_t pos = order[i];
                std::string txt = (pos < ids.size()) ? loaded->tokenizer.get_token(static_cast<int>(ids[pos])) : "?";
                std::cout << " [" << pos << ":\"" << tok_preview(txt) << "\" " << attn[pos] << "]";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    return 0;
}

}  // namespace cli
}  // namespace serius
