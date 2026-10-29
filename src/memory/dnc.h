#pragma once

#include "../tensor/tensor.h"

#include <vector>

namespace serius {
namespace memory {

class DNCMemory
{
   public:
    DNCMemory(size_t n_slots, size_t d_mem);

    void write(const core::Tensor &write_key, const core::Tensor &write_vector,
               float write_strength, float free_gate);
    core::Tensor read(const core::Tensor &read_key, float read_strength) const;

    const core::Tensor &matrix() const { return M_; }

   private:
    size_t n_;
    size_t d_;
    core::Tensor M_;      // [n, d]
    core::Tensor keys_;   // [n, d]
    core::Tensor usage_;  // [n]
};

}  // namespace memory
}  // namespace serius