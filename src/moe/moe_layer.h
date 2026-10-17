#pragma once

#include <memory>
#include <vector>

#include "../autograd/node.h"
#include "expert.h"
#include "router.h"

namespace serius {
namespace model {

class MoELayer
{
   public:
    MoELayer(size_t d_model, size_t num_experts, size_t top_k, size_t shared_experts = 1, unsigned seed = 42,
             size_t expert_d_ff = 0);
    ~MoELayer();

    core::autograd::Node *forward(core::autograd::Node *input) const;
    std::vector<core::autograd::Node *> parameters() const;

    const Router &router() const { return router_; }
    size_t num_experts() const { return experts_.size(); }
    size_t top_k() const { return router_.top_k(); }
    size_t shared_experts() const { return shared_experts_; }

   private:
    Router router_;
    size_t shared_experts_;
    std::vector<std::unique_ptr<Expert>> shared_expert_layers_;
    std::vector<std::unique_ptr<Expert>> experts_;
};

}  // namespace model
}  // namespace serius
