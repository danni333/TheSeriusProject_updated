#pragma once

#include "transport.h"

#include <string>

namespace serius {
namespace mcp {

/// HTTP + SSE transport to a remote MCP server.
/// Requires an HTTP client library (e.g. cpp-httplib / curl) which is not a
/// hard dependency of the core engine; this header declares the interface and
/// a minimal stub returning "not implemented".
class HttpTransport : public Transport
{
   public:
    explicit HttpTransport(const std::string &endpoint);
    ~HttpTransport();

    HttpTransport(const HttpTransport &) = delete;
    HttpTransport &operator=(const HttpTransport &) = delete;

    json send_request(const std::string &method,
                      const json &params = json::object()) override;

   private:
    std::string endpoint_;
};

}  // namespace mcp
}  // namespace serius
