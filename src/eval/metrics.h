#pragma once

#include <string>
#include <vector>

namespace serius {
namespace eval {

struct TrainingMetrics
{
    std::vector<int> steps;
    std::vector<float> train_loss;
    std::vector<float> val_loss;

    /// Overfitting detection: train loss decreasing while val loss increasing.
    bool is_overfitting(int window = 100) const;

    void export_csv(const std::string &path) const;
};

}  // namespace eval
}  // namespace serius
