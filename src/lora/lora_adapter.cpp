#include "lora_adapter.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

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

    std::string header_str(header_len, '\0');
    in.read(header_str.data(), static_cast<std::streamsize>(header_len));

    auto meta = nlohmann::json::parse(header_str);

    // Read tensor data for each layer
    auto &layer_info = meta["layers"];
    for (const auto &entry : layer_info) {
        std::string name = entry["name"];
        auto it = layers_.find(name);
        if (it == layers_.end()) continue;

        auto &layer = it->second;

        size_t a_bytes = entry["A_bytes"];
        size_t b_bytes = entry["B_bytes"];

        in.read(reinterpret_cast<char *>(layer->A().data.data()), static_cast<std::streamsize>(a_bytes));
        in.read(reinterpret_cast<char *>(layer->B().data.data()), static_cast<std::streamsize>(b_bytes));
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
