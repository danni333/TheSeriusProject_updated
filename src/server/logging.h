#pragma once

#include <string>

namespace serius {
namespace server {

struct RequestInfo
{
    std::string id;
    std::string model;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    float ttft_ms = 0.0f;
    float total_ms = 0.0f;
    std::string finish_reason;
};

/// Structured JSON log line for a completed request.
std::string log_request_json(const RequestInfo &req);

}  // namespace server
}  // namespace serius
