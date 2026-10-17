#include "export.h"

#include <sstream>

namespace serius {
namespace interpret {

std::string export_attention_json(const AttentionCapture &cap, const std::vector<std::string> &tokens)
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"tokens\": [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "\"" << tokens[i] << "\"";
    }
    oss << "],\n";

    oss << "  \"layers\": [\n";
    for (size_t l = 0; l < cap.num_layers(); ++l) {
        if (l > 0) oss << ",\n";
        oss << "    {\n";
        oss << "      \"layer\": " << l << ",\n";
        oss << "      \"heads\": [\n";
        for (size_t h = 0; h < cap.num_heads(static_cast<int>(l)); ++h) {
            if (h > 0) oss << ",\n";
            oss << "        [\n";
            Tensor attn = cap.get(static_cast<int>(l), static_cast<int>(h));
            if (attn.size() > 0) {
                size_t seq = attn.shape()[0];
                for (size_t i = 0; i < seq; ++i) {
                    if (i > 0) oss << ",\n";
                    oss << "          [";
                    for (size_t j = 0; j < seq; ++j) {
                        if (j > 0) oss << ", ";
                        oss << attn[i * seq + j];
                    }
                    oss << "]";
                }
            }
            oss << "\n        ]";
        }
        oss << "\n      ]\n";
        oss << "    }";
    }
    oss << "\n  ]\n";
    oss << "}\n";
    return oss.str();
}

std::string export_attention_csv(const AttentionCapture &cap, int layer, int head)
{
    std::ostringstream oss;
    Tensor attn = cap.get(layer, head);
    if (attn.size() == 0) return "";

    size_t seq = attn.shape()[0];
    // Header
    oss << "token";
    for (size_t j = 0; j < seq; ++j) {
        oss << ",t" << j;
    }
    oss << "\n";

    for (size_t i = 0; i < seq; ++i) {
        oss << "t" << i;
        for (size_t j = 0; j < seq; ++j) {
            oss << "," << attn[i * seq + j];
        }
        oss << "\n";
    }
    return oss.str();
}

std::string export_probe_json(const ActivationProbe &probe)
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"routing\": [\n";

    bool first = true;
    for (const auto &entry : probe.routing_log()) {
        if (!first) oss << ",\n";
        first = false;
        oss << "    {\n";
        oss << "      \"layer\": " << entry.layer << ",\n";
        oss << "      \"token_pos\": " << entry.token_pos << ",\n";
        oss << "      \"experts\": [";
        for (size_t i = 0; i < entry.experts.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << entry.experts[i];
        }
        oss << "],\n";
        oss << "      \"weights\": [";
        for (size_t i = 0; i < entry.weights.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << entry.weights[i];
        }
        oss << "]\n";
        oss << "    }";
    }
    oss << "\n  ]\n";
    oss << "}\n";
    return oss.str();
}

}  // namespace interpret
}  // namespace serius