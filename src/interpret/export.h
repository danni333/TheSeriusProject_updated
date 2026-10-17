#pragma once

#include <string>

#include "attention_capture.h"
#include "activation_probe.h"

namespace serius {
namespace interpret {

/// Export captured attention weights to JSON format (BertViz-compatible)
std::string export_attention_json(const AttentionCapture &cap, const std::vector<std::string> &tokens);

/// Export captured attention weights to CSV heatmap format
std::string export_attention_csv(const AttentionCapture &cap, int layer, int head);

/// Export activation probe data to JSON
std::string export_probe_json(const ActivationProbe &probe);

}  // namespace interpret
}  // namespace serius