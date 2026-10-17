#include "logging.h"

#include <sstream>

namespace serius {
namespace server {

std::string log_request_json(const RequestInfo &req)
{
    std::ostringstream oss;
    oss << "{";
    oss << "\"request_id\":\"" << req.id << "\",";
    oss << "\"model\":\"" << req.model << "\",";
    oss << "\"prompt_tokens\":" << req.prompt_tokens << ",";
    oss << "\"completion_tokens\":" << req.completion_tokens << ",";
    oss << "\"ttft_ms\":" << req.ttft_ms << ",";
    oss << "\"total_ms\":" << req.total_ms << ",";
    oss << "\"finish_reason\":\"" << req.finish_reason << "\"";
    oss << "}";
    return oss.str();
}

}  // namespace server
}  // namespace serius
