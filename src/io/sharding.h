#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "safetensors.h"

namespace serius {
namespace io {

struct TensorShardInfo
{
    std::string shard_file;
    size_t bytes = 0;
};

struct ShardedSafetensorsIndex
{
    std::map<std::string, TensorShardInfo> tensors;
    std::map<std::string, std::string> metadata;
    std::string index_path;
};

/// Load a sharded safetensors index manifest and materialize all tensors in memory.
bool load_sharded_safetensors(const std::string &index_path, std::map<std::string, core::Tensor> *out);

/// Write a sharded safetensors index for a tensor collection.
bool write_sharded_safetensors_index(const std::string &directory,
                                     const std::string &base_name,
                                     const std::vector<std::pair<std::string, core::Tensor>> &tensors,
                                     size_t max_shard_bytes = 0);

}  // namespace io
}  // namespace serius
