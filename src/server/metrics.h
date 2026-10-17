#pragma once

#include <atomic>
#include <string>

namespace serius {
namespace server {

/// Inference metrics exposed in Prometheus text format (no external prometheus-cpp dependency).
class InferenceMetrics
{
   public:
    InferenceMetrics() = default;

    void on_request_start();
    void on_request_end(float duration_s);
    void on_token_generated(float latency_ms);

    /// Export all metrics in Prometheus exposition format.
    std::string prometheus_text() const;

   private:
    std::atomic<long long> requests_total_{0};
    std::atomic<long long> tokens_generated_{0};
    std::atomic<int> active_requests_{0};
};

}  // namespace server
}  // namespace serius
