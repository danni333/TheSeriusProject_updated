#include "metrics.h"

#include <sstream>

namespace serius {
namespace server {

void InferenceMetrics::on_request_start()
{
    ++requests_total_;
    ++active_requests_;
}

void InferenceMetrics::on_request_end(float /*duration_s*/)
{
    --active_requests_;
}

void InferenceMetrics::on_token_generated(float /*latency_ms*/)
{
    ++tokens_generated_;
}

std::string InferenceMetrics::prometheus_text() const
{
    std::ostringstream oss;
    oss << "# HELP llm_requests_total Total number of inference requests\n";
    oss << "# TYPE llm_requests_total counter\n";
    oss << "llm_requests_total " << requests_total_.load() << "\n";
    oss << "# HELP llm_tokens_generated_total Total number of generated tokens\n";
    oss << "# TYPE llm_tokens_generated_total counter\n";
    oss << "llm_tokens_generated_total " << tokens_generated_.load() << "\n";
    oss << "# HELP llm_active_requests Number of currently active requests\n";
    oss << "# TYPE llm_active_requests gauge\n";
    oss << "llm_active_requests " << active_requests_.load() << "\n";
    return oss.str();
}

}  // namespace server
}  // namespace serius
