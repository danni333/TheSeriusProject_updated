#pragma once

#include <string>

namespace serius {
namespace model {
class TransformerModel;
}

namespace io {

/// Save all model weights to a single binary file (FP32, no quantization).
/// Format: 4-byte magic "SERW", then raw floats in the same order as
/// TransformerModel::parameters() (embedding, then per-block attn/ln/ffn, then
/// final_norm, output_proj). Use for small/simple models (~3–4 GB) and benchmarks.
/// Returns true on success.
bool save_simple_weights(const std::string &path, const model::TransformerModel &model);

/// Load weights from a file produced by save_simple_weights. The model must
/// already be constructed with the same ModelConfig as the saved model;
/// otherwise shapes will mismatch and load returns false.
/// Returns true if all parameter tensors were filled successfully.
bool load_simple_weights(const std::string &path, model::TransformerModel *model);

}  // namespace io
}  // namespace serius
