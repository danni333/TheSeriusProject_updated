#include "safetensors_writer.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace serius {
namespace io {

bool write_sharded_safetensors(const std::string &directory,
                               const std::string &base_name,
                               const std::vector<std::pair<std::string, core::Tensor>> &tensors,
                               size_t max_shard_bytes)
{
    if (tensors.empty()) {
        return false;
    }

    if (!fs::exists(directory)) {
        fs::create_directories(directory);
    }

    constexpr size_t kDefaultShardBytes = 5ULL * 1024ULL * 1024ULL * 1024ULL;
    const size_t shard_limit = max_shard_bytes == 0 ? kDefaultShardBytes : max_shard_bytes;

    std::vector<std::pair<std::string, core::Tensor>> current_shard;
    std::vector<std::string> shard_files;
    size_t shard_bytes = 0;
    size_t shard_index = 0;

    nlohmann::json index = nlohmann::json::object();
    index["__metadata__"] = { {"format", "serius-sharded"}, {"version", "1.0"} };
    index["tensors"] = nlohmann::json::object();

    auto flush_shard = [&](const std::vector<std::pair<std::string, core::Tensor>> &list) {
        if (list.empty()) {
            return;
        }

        std::string shard_name = base_name + "." + std::to_string(++shard_index) + ".safetensors";
        std::string shard_path = (fs::path(directory) / shard_name).string();
        SafetensorsWriter writer;
        for (const auto &[name, tensor] : list) {
            writer.add_tensor(name, tensor);
        }
        writer.save(shard_path);
        shard_files.push_back(shard_name);

        for (const auto &[name, tensor] : list) {
            index["tensors"][name] = { {"file", shard_name}, {"bytes", tensor.size() * tensor.element_size()} };
        }
    };

    for (const auto &[name, tensor] : tensors) {
        const size_t tensor_bytes = tensor.size() * tensor.element_size();
        if (!current_shard.empty() && shard_bytes + tensor_bytes > shard_limit) {
            flush_shard(current_shard);
            current_shard.clear();
            shard_bytes = 0;
        }

        current_shard.emplace_back(name, tensor);
        shard_bytes += tensor_bytes;
    }

    if (!current_shard.empty()) {
        flush_shard(current_shard);
    }

    const std::string index_path = (fs::path(directory) / (base_name + ".safetensors.index.json")).string();
    std::ofstream out(index_path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << index.dump(2);
    return true;
}

}  // namespace io
}  // namespace serius
