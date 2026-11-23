#include "lora_adapter.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace serius {
namespace lora {

LoRAAdapter::LoRAAdapter(const LoRAConfig &config) : config_(config) {}

LoRALinear &LoRAAdapter::add_layer(const std::string &name, core::autograd::Node &weight)
{
    float alpha = config_.alpha > 0 ? config_.alpha : static_cast<float>(config_.rank);
    auto layer = std::make_unique<LoRALinear>(weight, config_.rank, alpha);
    auto *ptr = layer.get();
    layers_[name] = std::move(layer);
    return *ptr;
}

bool LoRAAdapter::should_adapt(const std::string &layer_name) const
{
    for (const auto &target : config_.target_modules) {
        if (layer_name.find(target) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::vector<core::autograd::Node *> LoRAAdapter::parameters() const
{
    std::vector<core::autograd::Node *> params;
    for (const auto &[name, layer] : layers_) {
        auto lp = const_cast<LoRALinear *>(layer.get())->parameters();
        params.insert(params.end(), lp.begin(), lp.end());
    }
    return params;
}

LoRALinear *LoRAAdapter::get_layer(const std::string &name)
{
    auto it = layers_.find(name);
    return it != layers_.end() ? it->second.get() : nullptr;
}

const LoRALinear *LoRAAdapter::get_layer(const std::string &name) const
{
    auto it = layers_.find(name);
    return it != layers_.end() ? it->second.get() : nullptr;
}

void LoRAAdapter::merge_all()
{
    for (auto &[name, layer] : layers_) {
        layer->merge();
    }
}

void LoRAAdapter::unmerge_all()
{
    for (auto &[name, layer] : layers_) {
        layer->unmerge();
    }
}

void LoRAAdapter::save(const std::string &path) const
{
    // Binary format:
    // [JSON header with metadata]\n
    // [raw tensor data for each A, B]

    nlohmann::json meta;
    meta["rank"] = config_.rank;
    meta["alpha"] = config_.alpha;
    meta["target_modules"] = config_.target_modules;
    meta["num_layers"] = layers_.size();

    nlohmann::json layer_info = nlohmann::json::array();
    size_t data_offset = 0;

    for (const auto &[name, layer] : layers_) {
        nlohmann::json entry;
        entry["name"] = name;
        entry["rank"] = layer->rank();
        entry["alpha"] = layer->alpha();

        // A tensor info
        entry["A_shape"] = std::vector<size_t>(layer->A().data.shape());
        entry["A_offset"] = data_offset;
        entry["A_bytes"] = layer->A().data.size() * sizeof(float);
        data_offset += layer->A().data.size() * sizeof(float);

        // B tensor info
        entry["B_shape"] = std::vector<size_t>(layer->B().data.shape());
        entry["B_offset"] = data_offset;
        entry["B_bytes"] = layer->B().data.size() * sizeof(float);
        data_offset += layer->B().data.size() * sizeof(float);

        layer_info.push_back(entry);
    }
    meta["layers"] = layer_info;

    std::string header = meta.dump();

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("LoRAAdapter::save: cannot open: " + path);

    // Write header length + header
    uint64_t header_len = header.size();
    out.write(reinterpret_cast<const char *>(&header_len), sizeof(uint64_t));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    // Write tensor data
    for (const auto &[name, layer] : layers_) {
        out.write(reinterpret_cast<const char *>(layer->A().data.data()),
                  static_cast<std::streamsize>(layer->A().data.size() * sizeof(float)));
        out.write(reinterpret_cast<const char *>(layer->B().data.data()),
                  static_cast<std::streamsize>(layer->B().data.size() * sizeof(float)));
    }
}

void LoRAAdapter::load(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("LoRAAdapter::load: cannot open: " + path);

    // Read header
    uint64_t header_len = 0;
    in.read(reinterpret_cast<char *>(&header_len), sizeof(uint64_t));
    if (!in || header_len > (64ull << 20)) {
        throw std::runtime_error("LoRAAdapter::load: invalid header length in " + path);
    }

    std::string header_str(header_len, '\0');
    in.read(header_str.data(), static_cast<std::streamsize>(header_len));
    if (!in) throw std::runtime_error("LoRAAdapter::load: truncated header in " + path);

    auto meta = nlohmann::json::parse(header_str);
    if (!meta.contains("layers") || !meta["layers"].is_array()) {
        throw std::runtime_error("LoRAAdapter::load: missing layers array in " + path);
    }

    // Tensor data starts right after the header; offsets in the metadata are
    // relative to this position.
    const std::streampos data_start = in.tellg();

    // Read tensor data for each layer
    for (const auto &entry : meta["layers"]) {
        std::string name = entry.value("name", "");
        auto it = layers_.find(name);
        if (it == layers_.end()) continue;

        auto &layer = it->second;

        const size_t a_expected = layer->A().data.size() * sizeof(float);
        const size_t b_expected = layer->B().data.size() * sizeof(float);
        const size_t a_bytes = entry.value("A_bytes", size_t(0));
        const size_t b_bytes = entry.value("B_bytes", size_t(0));
        if (a_bytes != a_expected || b_bytes != b_expected) {
            throw std::runtime_error("LoRAAdapter::load: size mismatch for layer '" + name + "' in " + path);
        }

        const uint64_t a_off = entry.value("A_offset", uint64_t(0));
        const uint64_t b_off = entry.value("B_offset", uint64_t(0));

        layer->A().data.syncToHost();
        in.seekg(data_start + static_cast<std::streamoff>(a_off));
        if (!in) throw std::runtime_error("LoRAAdapter::load: bad A offset for '" + name + "'");
        in.read(reinterpret_cast<char *>(layer->A().data.data()), static_cast<std::streamsize>(a_bytes));
        if (!in) throw std::runtime_error("LoRAAdapter::load: truncated A data for '" + name + "'");
        if (layer->A().data.is_gpu()) {
            layer->A().data.mark_dirty(core::Tensor::DataLocation::CPU);
        }

        layer->B().data.syncToHost();
        in.seekg(data_start + static_cast<std::streamoff>(b_off));
        if (!in) throw std::runtime_error("LoRAAdapter::load: bad B offset for '" + name + "'");
        in.read(reinterpret_cast<char *>(layer->B().data.data()), static_cast<std::streamsize>(b_bytes));
        if (!in) throw std::runtime_error("LoRAAdapter::load: truncated B data for '" + name + "'");
        if (layer->B().data.is_gpu()) {
            layer->B().data.mark_dirty(core::Tensor::DataLocation::CPU);
        }
    }
}

size_t LoRAAdapter::num_trainable_params() const
{
    size_t total = 0;
    for (const auto &[name, layer] : layers_) {
        total += layer->A().data.size() + layer->B().data.size();
    }
    return total;
}

size_t LoRAAdapter::trainable_bytes() const
{
    return num_trainable_params() * sizeof(float);
}

}  // namespace lora
}  // namespace serius
