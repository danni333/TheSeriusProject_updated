#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "../tensor/tensor.h"


namespace serius {
namespace core {
namespace autograd {
struct Node;
}
}  // namespace core

namespace debug {

struct TensorStats
{
    std::string name;
    int step;
    float mean;
    float std_dev;
    float min_val;
    float max_val;
    bool has_nan;
    bool has_inf;
    float sparsity;  // fraction of zeros
    float norm;      // L2 norm for gradient checking
    float max_abs;   // maximum absolute value
    int64_t size;    // tensor size in bytes
};

class TensorLogger
{
   public:
    enum Level { SILENT, SUMMARY, FULL, PANIC };

    void set_level(Level level) { level_ = level; }
    Level level() const { return level_; }

    /// Log tensor stats — triggers panic if NaN/Inf detected in PANIC mode
    void log(const core::Tensor &tensor, const std::string &name, int step);

    /// Export all logged stats to CSV
    void export_csv(const std::string &path) const;

    /// Export all logged stats to JSON
    void export_json(const std::string &path) const;

    /// Specific check for anomalous gradients (e.g. dying ReLU or exploding grad)
    void log_gradients(const std::vector<core::autograd::Node *> &parameters, int step);

    /// Get the log history
    const std::vector<TensorStats> &history() const { return history_; }

    /// Clear history
    void clear() { history_.clear(); }

    /// Check if any tensor has NaN/Inf in the history
    bool has_anomalies() const;

    /// Get statistics summary
    std::string get_summary() const;

   private:
    Level level_ = SUMMARY;
    std::vector<TensorStats> history_;
    std::chrono::high_resolution_clock::time_point start_time_;

    TensorStats compute(const core::Tensor &tensor, const std::string &name, int step) const;
    void panic(const TensorStats &stats) const;
    void dump_tensor(const core::Tensor &tensor, const std::string &name, int step) const;
};

/// Singleton access
TensorLogger &get_tensor_logger();

}  // namespace debug
}  // namespace serius
