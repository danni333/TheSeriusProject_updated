#include "sharding.h"

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace serius {
namespace io {

bool load_sharded_safetensors(const std::string &index_path, std::map<std::string, core::Tensor> *out)
{
    if (!out) {
        return false;
    }
    out->clear();

    std::ifstream in(index_path, std::ios::binary);
    if (!in) {
        return false;
    }

    try {
        const auto index = nlohmann::json::parse(in);
        const auto root = index.at("tensors");
        const fs::path base_dir = fs::path(index_path).parent_path();

        for (auto it = root.begin(); it != root.end(); ++it) {
            const std::string name = it.key();
            const auto &entry = it.value();
            const std::string shard_name = entry.at("file").get<std::string>();
            const fs::path shard_path = base_dir / shard_name;
            SafetensorsReader reader(shard_path.string());
            if (!reader.has_tensor(name)) {
                continue;
            }
            (*out)[name] = reader.get_tensor(name);
        }
        return !out->empty();
    } catch (const std::exception &) {
        return false;
    }
}

bool write_sharded_safetensors_index(const std::string &directory,
                                     const std::string &base_name,
                                     const std::vector<std::pair<std::string, core::Tensor>> &tensors,
                                     size_t max_shard_bytes)
{
    if (tensors.empty()) {
        return false;
    }

    const fs::path dir(directory);
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }

    std::vector<std::pair<std::string, core::Tensor>> current;
    size_t current_bytes = 0;
    constexpr size_t kDefaultShardBytes = 5ULL * 1024ULL * 1024ULL * 1024ULL;
    const size_t shard_limit = max_shard_bytes == 0 ? kDefaultShardBytes : max_shard_bytes;

    nlohmann::json index = nlohmann::json::object();
    index["__metadata__"] = { {"format", "serius-sharded"}, {"version", "1.0"} };
    index["tensors"] = nlohmann::json::object();

    auto flush = [&](const std::vector<std::pair<std::string, core::Tensor>> &chunk) {
        if (chunk.empty()) {
            return;
        }

        const std::string shard_name = base_name + "." + std::to_string(index["_shard_count"].get<size_t>() + 1) + ".safetensors";
        const std::string shard_path = (dir / shard_name).string();
        SafetensorsWriter writer;
        for (const auto &[tensor_name, tensor] : chunk) {
            writer.add_tensor(tensor_name, tensor);
        }
        writer.save(shard_path);

        for (const auto &[tensor_name, tensor] : chunk) {
            index["tensors"][tensor_name] = { {"file", shard_name}, {"bytes", tensor.size() * tensor.element_size()} };
        }
        index["_shard_count"] = index["_shard_count"].get<size_t>() + 1;
    };

    index["_shard_count"] = 0;
    for (const auto &[tensor_name, tensor] : tensors) {
        const size_t tensor_bytes = tensor.size() * tensor.element_size();
        if (!current.empty() && current_bytes + tensor_bytes > shard_limit) {
            flush(current);
            current.clear();
            current_bytes = 0;
        }
        current.emplace_back(tensor_name, tensor);
        current_bytes += tensor_bytes;
    }
    if (!current.empty()) {
        flush(current);
    }

    const std::string index_path = (dir / (base_name + ".safetensors.index.json")).string();
    std::ofstream out(index_path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << index.dump(2);
    return true;
}

}  // namespace io
}  // namespace serius
