#include "user_adapter.h"

#include <fstream>
#include <stdexcept>

namespace serius {
namespace federated {

UserAdapter::UserAdapter(std::string user_id, lora::LoRAConfig config)
    : user_id_(std::move(user_id)), adapter_(config)
{
}

void UserAdapter::save() const
{
    if (local_path_.empty()) {
        throw std::runtime_error("UserAdapter::save: local_path not set");
    }
    adapter_.save(local_path_);
}

void UserAdapter::load()
{
    if (local_path_.empty()) {
        throw std::runtime_error("UserAdapter::load: local_path not set");
    }
    adapter_.load(local_path_);
}

}  // namespace federated
}  // namespace serius
