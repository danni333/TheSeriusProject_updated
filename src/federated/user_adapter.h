#pragma once

#include <string>

#include "../lora/lora_adapter.h"

namespace serius {
namespace federated {

/// Per-user LoRA adapter for on-device personalization.
/// Stores a user-specific LoRA adapter that can be fine-tuned locally
/// and uploaded/downloaded as noisy delta updates.
class UserAdapter
{
   public:
    explicit UserAdapter(std::string user_id,
                          lora::LoRAConfig config = lora::LoRAConfig());

    const std::string &user_id() const { return user_id_; }
    lora::LoRAAdapter &adapter() { return adapter_; }
    const lora::LoRAAdapter &adapter() const { return adapter_; }

    size_t num_layers() const { return adapter_.num_layers(); }

    void set_local_path(std::string path) { local_path_ = std::move(path); }
    const std::string &local_path() const { return local_path_; }

    /// Save the adapter weights to local_path.
    void save() const;

    /// Load the adapter weights from local_path.
    void load();

   private:
    std::string user_id_;
    lora::LoRAAdapter adapter_;
    std::string local_path_;
};

}  // namespace federated
}  // namespace serius
