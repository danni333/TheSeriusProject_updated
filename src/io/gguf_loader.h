#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

#include "../tensor/tensor.h"
#include "../transformer/model.h"

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

GGUFMeta load_gguf_metadata(const std::string &path);

std::vector<std::string> gguf_tensor_names(const std::string &path);

std::optional<GGUFTensorInfo> get_tensor_info(const std::string &path, const std::string &tensor_name);

std::unordered_map<std::string, core::Tensor> load_all_tensors(const std::string &path);

serius::model::ModelConfig gguf_to_model_config(const std::string &path);

}  // namespace io
}  // namespace serius
