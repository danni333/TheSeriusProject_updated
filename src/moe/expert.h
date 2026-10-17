#pragma once

#include <memory>
#include <vector>

#include "../tensor/tensor.h"
#include "../transformer/ffn.h"

namespace serius {
namespace model {

class Expert
{
   public:
    Expert(size_t d_model, size_t d_ff, unsigned seed = 42);
    ~Expert();

    core::Tensor forward(const core::Tensor &input) const;
    core::autograd::Node *forward(core::autograd::Node *input) const;
    std::vector<core::autograd::Node *> parameters() const;

    size_t d_model() const { return d_model_; }
    size_t d_ff() const { return d_ff_; }

   private:
    size_t d_model_;
    size_t d_ff_;
    std::unique_ptr<FFN> ffn_;
};

}  // namespace model
}  // namespace serius
