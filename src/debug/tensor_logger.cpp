#include "tensor_logger.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace serius {
namespace debug {

TensorStats TensorLogger::compute(const core::Tensor &tensor, const std::string &name, int step) const
{
    TensorStats stats{};
    stats.name = name;
    stats.step = step;
    stats.has_nan = false;
    stats.has_inf = false;

    if (tensor.size() == 0) {
        stats.mean = 0.0f;
        stats.std_dev = 0.0f;
        stats.min_val = 0.0f;
        stats.max_val = 0.0f;
        stats.sparsity = 1.0f;
        return stats;
    }

    const float *data = tensor.data();
    size_t size = tensor.size();

    double sum = 0.0;
    float min_v = data[0];
    float max_v = data[0];
    size_t zero_count = 0;

    for (size_t i = 0; i < size; ++i) {
        float val = data[i];
        if (std::isnan(val)) {
            stats.has_nan = true;
        }
        if (std::isinf(val)) {
            stats.has_inf = true;
        }
        sum += static_cast<double>(val);
        min_v = std::min(min_v, val);
        max_v = std::max(max_v, val);
        if (val == 0.0f) {
            zero_count++;
        }
    }

    stats.mean = static_cast<float>(sum / static_cast<double>(size));
    stats.min_val = min_v;
    stats.max_val = max_v;
    stats.sparsity = static_cast<float>(zero_count) / static_cast<float>(size);

    // Compute std dev
    double var_sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = static_cast<double>(data[i]) - static_cast<double>(stats.mean);
        var_sum += diff * diff;
    }
    stats.std_dev = static_cast<float>(std::sqrt(var_sum / static_cast<double>(size)));

    return stats;
}

void TensorLogger::panic(const TensorStats &stats) const
{
    std::cerr << "[TENSOR PANIC] " << stats.name << " @ step " << stats.step;
    if (stats.has_nan) {
        std::cerr << " — contains NaN!";
    }
    if (stats.has_inf) {
        std::cerr << " — contains Inf!";
    }
    std::cerr << " (mean=" << stats.mean << ", std=" << stats.std_dev << ", min=" << stats.min_val
              << ", max=" << stats.max_val << ")" << std::endl;
}

void TensorLogger::log(const core::Tensor &tensor, const std::string &name, int step)
{
    if (level_ == SILENT) {
        return;
    }

    TensorStats stats = compute(tensor, name, step);
    history_.push_back(stats);

    if (level_ >= SUMMARY) {
        std::cout << "[TensorLog] " << name << " step=" << step << " mean=" << std::fixed << std::setprecision(6)
                  << stats.mean << " std=" << stats.std_dev << " min=" << stats.min_val << " max=" << stats.max_val;
        if (stats.has_nan) {
            std::cout << " [NaN!]";
        }
        if (stats.has_inf) {
            std::cout << " [Inf!]";
        }
        std::cout << std::endl;
    }

    if (level_ == PANIC && (stats.has_nan || stats.has_inf)) {
        panic(stats);
        throw std::runtime_error("Tensor health check failed: NaN or Inf detected in '" + name + "'");
    }
}

void TensorLogger::export_csv(const std::string &path) const
{
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + path);
    }

    file << "name,step,mean,std,min,max,has_nan,has_inf,sparsity\n";
    for (const auto &stats : history_) {
        file << stats.name << "," << stats.step << "," << std::fixed << std::setprecision(8) << stats.mean << ","
             << stats.std_dev << "," << stats.min_val << "," << stats.max_val << "," << (stats.has_nan ? 1 : 0) << ","
             << (stats.has_inf ? 1 : 0) << "," << stats.sparsity << "\n";
    }
}

TensorLogger &get_tensor_logger()
{
    static TensorLogger instance;
    return instance;
}

}  // namespace debug
}  // namespace serius
