#include "nan_debugger.h"

#include <cmath>
#include <cstring>

namespace serius {
namespace debug {

bool has_nan_or_inf(const core::Tensor &t)
{
    const float *ptr = t.data();
    if (ptr == nullptr) {
        return false;
    }
    for (size_t i = 0; i < t.size(); ++i) {
        float v = ptr[i];
        if (std::isnan(v) || std::isinf(v)) {
            return true;
        }
    }
    return false;
}

std::string find_first_nan(const std::vector<NamedTensor> &tensors)
{
    for (const auto &[name, ptr] : tensors) {
        if (ptr != nullptr && has_nan_or_inf(*ptr)) {
            return name;
        }
    }
    return {};
}

}  // namespace debug
}  // namespace serius
