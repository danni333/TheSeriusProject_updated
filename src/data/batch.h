#pragma once

#include "../tensor/tensor.h"

namespace serius {

struct Batch
{
    core::Tensor input_ids;
    core::Tensor target_ids;
    core::Tensor padding_mask;

    Batch(core::Tensor input, core::Tensor target, core::Tensor mask)
        : input_ids(std::move(input)), target_ids(std::move(target)), padding_mask(std::move(mask))
    {
    }
};

}  // namespace serius
