#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../quant/quant_weight.h"
#include "../tensor/tensor.h"
#include "../transformer/model.h"

namespace serius {
namespace io {

// Values match ggml.h GGML_TYPE_* exactly — do not reorder.
enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    F8_E4M3 = 43
};

struct GGUFMeta
{
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    bool valid = false;
};

struct GGUFTensorInfo
{
    std::string name;
    std::vector<uint64_t> shape;
    GGMLType dtype;
    uint64_t offset;
};

/// Vocabulary and merge rules extracted from a GGUF file's KV metadata.
struct GGUFTokenizerData
{
    std::string tokenizer_model;       ///< "llama", "gpt2", etc.
    std::vector<std::string> tokens;   ///< token strings, index = token-id
    std::vector<int32_t> token_types;  ///< 1=normal, 3=control, 6=byte, etc.
    std::vector<float> scores;         ///< SentencePiece log-prob scores
    std::vector<std::string> merges;   ///< BPE merge rules: "▁Ġ a" etc.
    int32_t bos_token_id = 1;
    int32_t eos_token_id = 2;
    bool valid = false;
};

GGUFMeta load_gguf_metadata(const std::string &path);

std::vector<std::string> gguf_tensor_names(const std::string &path);

/// Aggregate weight statistics read in a single pass over the GGUF tensor
/// infos: total parameter count (sum of element counts across all tensors)
/// and total payload bytes in the data section. `valid` is false when the
/// file is not a readable GGUF.
struct GGUFWeightStats
{
    uint64_t total_params = 0;
    uint64_t total_bytes = 0;
    uint64_t tensors_counted = 0;
    bool valid = false;
};

GGUFWeightStats gguf_weight_stats(const std::string &path);

std::optional<GGUFTensorInfo> get_tensor_info(const std::string &path, const std::string &tensor_name);

std::unordered_map<std::string, core::Tensor> load_all_tensors(
    const std::string &path, std::unordered_map<std::string, quant::QuantWeight> *quant_out = nullptr);

/// Returns the value of `general.architecture` (e.g. "llama", "qwen2"), or "".
std::string gguf_architecture(const std::string &path);

serius::model::ModelConfig gguf_to_model_config(const std::string &path);

/// Extract vocabulary and merge rules from a GGUF file.
/// Use GGUFTokenizerData to populate a BPETokenizer via load_from_gguf().
GGUFTokenizerData load_gguf_tokenizer(const std::string &path);

/// Convert a native-Q4_K QuantWeight (raw superblocks) into an equivalent
/// Q4_1 QuantWeight — used when row-fusion requires a uniform quant type
/// (e.g. Q4_K q/k fused with a Q6_K-derived Q4_1 v). Returns an invalid
/// QuantWeight on mismatch.
quant::QuantWeight quant_weight_q4k_to_q41(const quant::QuantWeight &w);

/// Convert a native-Q8_0 QuantWeight into an equivalent Q4_1 QuantWeight —
/// used when row-fusion requires a uniform quant type (e.g. Q8_0 v fused
/// with Q4_1 q/k). Returns an invalid QuantWeight on mismatch.
quant::QuantWeight quant_weight_q80_to_q41(const quant::QuantWeight &w);

/// True when `w`'s packed bytes can be expanded to dense FP32 by
/// dequantize_quant_weight_dense (Q4_0/Q4_1/Q8_0/Q4_K/Q5_K GGUF row
/// layouts with block-aligned `in`).
bool quant_weight_dequantizable(const quant::QuantWeight &w);

/// Dequantize a packed QuantWeight to dense FP32 in GGUF row order
/// ([out][in] row-major — the same element order load_all_tensors produces
/// before the HF→CPP transpose). `out` must hold in*out floats.
/// Returns false on unsupported type or size mismatch.
bool dequantize_quant_weight_dense(const quant::QuantWeight &w, float *out);

}  // namespace io
}  // namespace serius
