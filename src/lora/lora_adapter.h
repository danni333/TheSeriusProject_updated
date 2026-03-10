#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../autograd/node.h"
#include "lora_linear.h"

namespace serius {
namespace lora {

/// Configuration for a LoRA adapter.
struct LoRAConfig
{
    int rank = 16;
    float alpha = 0.0f;  // 0 = use rank as alpha
    std::vector<std::string> target_modules = {"q_proj", "v_proj"};  // which layers to adapt
};

/// LoRA Adapter: manages a collection of LoRALinear layers applied to a model.
class LoRAAdapter
{
   public:
    explicit LoRAAdapter(const LoRAConfig &config = LoRAConfig());

    /// Apply LoRA to a named weight matrix.
    /// Returns a reference to the created LoRALinear.
    LoRALinear &add_layer(const std::string &name, core::autograd::Node &weight);

    /// Check if a layer name matches the target modules pattern.
    bool should_adapt(const std::string &layer_name) const;

    /// Get all trainable parameters across all LoRA layers.
    std::vector<core::autograd::Node *> parameters() const;

    /// Get a specific LoRA layer by name.
    LoRALinear *get_layer(const std::string &name);
    const LoRALinear *get_layer(const std::string &name) const;

    /// Merge all LoRA layers into the base weights.
    void merge_all();

    /// Un-merge all LoRA layers.
    void unmerge_all();

    /// Save LoRA weights to a file.
    void save(const std::string &path) const;

    /// Load LoRA weights from a file.
    /// The adapter must already have layers created (via add_layer).
    void load(const std::string &path);

    /// Get total number of trainable parameters.
    size_t num_trainable_params() const;

    /// Get total size in bytes of trainable parameters.
    size_t trainable_bytes() const;

    const LoRAConfig &config() const { return config_; }
    size_t num_layers() const { return layers_.size(); }

   private:
    LoRAConfig config_;
    std::map<std::string, std::unique_ptr<LoRALinear>> layers_;
};

}  // namespace lora
}  // namespace serius
