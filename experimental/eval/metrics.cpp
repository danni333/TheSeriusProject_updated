#include "metrics.h"

#include <algorithm>
#include <fstream>
#include <numeric>

namespace serius {
namespace eval {

namespace {
float trend(const std::vector<float> &values, int window)
{
    if (static_cast<int>(values.size()) < window) {
        return 0.0f;
    }
    const size_t start = values.size() - static_cast<size_t>(window);
    // Simple linear trend: (last - first) / window
    return (values.back() - values[start]) / static_cast<float>(window);
}
}  // namespace

bool TrainingMetrics::is_overfitting(int window) const
{
    if (val_loss.size() < static_cast<size_t>(window)) {
        return false;
    }
    const float train_trend = trend(train_loss, window);
    const float val_trend = trend(val_loss, window);
    return train_trend < -0.01f && val_trend > 0.01f;
}

void TrainingMetrics::export_csv(const std::string &path) const
{
    std::ofstream f(path);
    if (!f) {
        return;
    }
    f << "step,train_loss,val_loss\n";
    const size_t n = std::min({steps.size(), train_loss.size(), val_loss.size()});
    for (size_t i = 0; i < n; ++i) {
        f << steps[i] << "," << train_loss[i] << "," << val_loss[i] << "\n";
    }
}

}  // namespace eval
}  // namespace serius
