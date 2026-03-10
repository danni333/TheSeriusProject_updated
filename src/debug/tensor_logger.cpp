#include "tensor_logger.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "../autograd/node.h"


namespace serius {
namespace debug {

TensorStats TensorLogger::compute(const core::Tensor &tensor, const std::string &name, int step) const
{
    TensorStats stats{};
    stats.name = name;
    stats.step = step;
    stats.has_nan = false;
    stats.has_inf = false;
    stats.size = tensor.size() * sizeof(float);  // Approximate size in bytes

    if (tensor.size() == 0) {
        stats.mean = 0.0F;
        stats.std_dev = 0.0F;
        stats.min_val = 0.0F;
        stats.max_val = 0.0F;
        stats.sparsity = 1.0F;
        stats.norm = 0.0F;
        stats.max_abs = 0.0F;
        return stats;
    }

    const float *data = tensor.data();
    size_t size = tensor.size();

    double sum = 0.0;
    double sum_sq = 0.0;
    float min_v = data[0];
    float max_v = data[0];
    float max_abs_v = std::abs(data[0]);
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
        sum_sq += static_cast<double>(val) * static_cast<double>(val);
        min_v = std::min(min_v, val);
        max_v = std::max(max_v, val);
        max_abs_v = std::max(max_abs_v, std::abs(val));
        if (val == 0.0F) {
            zero_count++;
        }
    }

    stats.mean = static_cast<float>(sum / static_cast<double>(size));
    stats.min_val = min_v;
    stats.max_val = max_v;
    stats.max_abs = max_abs_v;
    stats.sparsity = static_cast<float>(zero_count) / static_cast<float>(size);

    double variance = (sum_sq / static_cast<double>(size)) - (stats.mean * stats.mean);
    stats.std_dev = static_cast<float>(std::sqrt(std::max(0.0, variance)));
    stats.norm = static_cast<float>(std::sqrt(sum_sq));

    return stats;
}

void TensorLogger::panic(const TensorStats &stats) const
{
    std::cerr << "\n=== TENSOR PANIC ===" << std::endl;
    std::cerr << "Tensor: " << stats.name << std::endl;
    std::cerr << "Step: " << stats.step << std::endl;
    std::cerr << "Stats: mean=" << stats.mean << " std=" << stats.std_dev << std::endl;
    std::cerr << "Range: [" << stats.min_val << ", " << stats.max_val << "]" << std::endl;
    std::cerr << "Max abs: " << stats.max_abs << std::endl;
    std::cerr << "L2 norm: " << stats.norm << std::endl;
    std::cerr << "Size: " << stats.size << " bytes" << std::endl;

    if (stats.has_nan) {
        std::cerr << "ERROR: Contains NaN!" << std::endl;
    }
    if (stats.has_inf) {
        std::cerr << "ERROR: Contains Inf!" << std::endl;
    }

    std::cerr << "===================\n" << std::endl;
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
                  << stats.mean << " std=" << stats.std_dev << " min=" << stats.min_val << " max=" << stats.max_val
                  << " norm=" << stats.norm;

        if (stats.has_nan) {
            std::cout << " [NaN!]";
        }
        if (stats.has_inf) {
            std::cout << " [Inf!]";
        }
        if (stats.sparsity > 0.9f) {
            std::cout << " [Sparse: " << (stats.sparsity * 100) << "% zeros]";
        }
        std::cout << std::endl;
    }

    if (level_ == PANIC && (stats.has_nan || stats.has_inf)) {
        panic(stats);
        dump_tensor(tensor, name, step);
        throw std::runtime_error("Tensor health check failed: NaN or Inf detected in '" + name + "'");
    }
}

void TensorLogger::export_csv(const std::string &path) const
{
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + path);
    }

    file << "name,step,mean,std,min,max,has_nan,has_inf,sparsity,norm,max_abs,size_bytes\n";
    for (const auto &stats : history_) {
        file << stats.name << "," << stats.step << "," << std::fixed << std::setprecision(8) << stats.mean << ","
             << stats.std_dev << "," << stats.min_val << "," << stats.max_val << "," << (stats.has_nan ? 1 : 0) << ","
             << (stats.has_inf ? 1 : 0) << "," << stats.sparsity << "," << stats.norm << "," << stats.max_abs << ","
             << stats.size << "\n";
    }
}

void TensorLogger::export_json(const std::string &path) const
{
    std::ostringstream oss;
    oss << "[\n";
    for (size_t i = 0; i < history_.size(); ++i) {
        const auto &stats = history_[i];
        oss << "  {\n";
        oss << "    \"name\": \"" << stats.name << "\",\n";
        oss << "    \"step\": " << stats.step << ",\n";
        oss << "    \"mean\": " << stats.mean << ",\n";
        oss << "    \"std\": " << stats.std_dev << ",\n";
        oss << "    \"min\": " << stats.min_val << ",\n";
        oss << "    \"max\": " << stats.max_val << ",\n";
        oss << "    \"has_nan\": " << (stats.has_nan ? "true" : "false") << ",\n";
        oss << "    \"has_inf\": " << (stats.has_inf ? "true" : "false") << ",\n";
        oss << "    \"sparsity\": " << stats.sparsity << ",\n";
        oss << "    \"norm\": " << stats.norm << ",\n";
        oss << "    \"max_abs\": " << stats.max_abs << ",\n";
        oss << "    \"size_bytes\": " << stats.size << "\n";
        oss << "  }";
        if (i < history_.size() - 1) {
            oss << ",\n";
        }
    }
    oss << "\n]";

    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    file << oss.str();
}

void TensorLogger::log_gradients(const std::vector<core::autograd::Node *> &parameters, int step)
{
    for (auto *param : parameters) {
        if (param->requires_grad && param->grad.size() > 0) {
            log(param->grad, param->name + ".grad", step);
        }
    }
}

bool TensorLogger::has_anomalies() const
{
    for (const auto &stats : history_) {
        if (stats.has_nan || stats.has_inf) {
            return true;
        }
    }
    return false;
}

std::string TensorLogger::get_summary() const
{
    std::ostringstream oss;
    oss << "TensorLogger Summary:\n";
    oss << "  Total logs: " << history_.size() << "\n";
    oss << "  Tensors with NaN: "
        << std::count_if(history_.begin(), history_.end(), [](const auto &s) { return s.has_nan; }) << "\n";
    oss << "  Tensors with Inf: "
        << std::count_if(history_.begin(), history_.end(), [](const auto &s) { return s.has_inf; }) << "\n";
    oss << "  Current level: ";
    switch (level_) {
        case SILENT:
            oss << "SILENT";
            break;
        case SUMMARY:
            oss << "SUMMARY";
            break;
        case FULL:
            oss << "FULL";
            break;
        case PANIC:
            oss << "PANIC";
            break;
    }
    oss << std::endl;
    return oss.str();
}

void TensorLogger::dump_tensor(const core::Tensor &tensor, const std::string &name, int step) const
{
    std::string filename = name + "_step" + std::to_string(step) + ".bin";
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to dump tensor to " << filename << std::endl;
        return;
    }

    const float *data = tensor.data();
    file.write(reinterpret_cast<const char *>(data), tensor.size() * sizeof(float));
    std::cerr << "Tensor dumped to " << filename << std::endl;
}

TensorLogger &get_tensor_logger()
{
    static TensorLogger instance;
    return instance;
}

}  // namespace debug
}  // namespace serius
