#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
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

    /// Get the log history
    const std::vector<TensorStats> &history() const { return history_; }

    /// Clear history
    void clear() { history_.clear(); }

   private:
    Level level_ = SUMMARY;
    std::vector<TensorStats> history_;

    TensorStats compute(const core::Tensor &tensor, const std::string &name, int step) const;
    void panic(const TensorStats &stats) const;
};

/// Singleton access
TensorLogger &get_tensor_logger();

}  // namespace debug
}  // namespace serius
