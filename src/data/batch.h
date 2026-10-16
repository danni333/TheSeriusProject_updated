#pragma once

#include "tensor/tensor.h"

namespace serius {

struct Batch
{
    Tensor input_ids;
    Tensor target_ids;
    Tensor padding_mask;
};

}  // namespace serius