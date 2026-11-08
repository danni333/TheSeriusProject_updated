#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

#include "../tensor/tensor.h"
#include "../transformer/model.h"
#include "../quant/quant_weight.h"

namespace serius {
namespace io {

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
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    BF16 = 21
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
    std::string              tokenizer_model;   ///< "llama", "gpt2", etc.
    std::vector<std::string> tokens;            ///< token strings, index = token-id
    std::vector<int32_t>     token_types;       ///< 1=normal, 3=control, 6=byte, etc.
    std::vector<float>       scores;            ///< SentencePiece log-prob scores
    std::vector<std::string> merges;            ///< BPE merge rules: "▁Ġ a" etc.
    int32_t bos_token_id = 1;
    int32_t eos_token_id = 2;
    bool valid = false;
};

GGUFMeta load_gguf_metadata(const std::string &path);

std::vector<std::string> gguf_tensor_names(const std::string &path);

std::optional<GGUFTensorInfo> get_tensor_info(const std::string &path, const std::string &tensor_name);

std::unordered_map<std::string, core::Tensor> load_all_tensors(
    const std::string &path,
    std::unordered_map<std::string, quant::QuantWeight> *quant_out = nullptr);

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

}  // namespace io
}  // namespace serius
