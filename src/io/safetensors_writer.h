#pragma once

#include <string>
#include <utility>
#include <vector>

#include "safetensors.h"

namespace serius {
namespace io {

/// Convenience API for writing a model as one or more .safetensors shard files.
/// The caller provides a collection of tensor-name / Tensor pairs; the function will
/// split them across shard files and emit a companion .safetensors.index.json manifest.
bool write_sharded_safetensors(const std::string &directory,
                               const std::string &base_name,
                               const std::vector<std::pair<std::string, core::Tensor>> &tensors,
                               size_t max_shard_bytes = 0);

}  // namespace io
}  // namespace serius
