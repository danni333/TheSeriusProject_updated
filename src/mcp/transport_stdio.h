#pragma once

#include "transport.h"

#include <cstdio>
#include <string>

namespace serius {
namespace mcp {

/// stdio transport: launches a subprocess and communicates over stdin/stdout.
/// Best-effort implementation; a real deployment should manage the subprocess
/// lifetime and buffered I/O carefully.
class StdioTransport : public Transport
{
   public:
    explicit StdioTransport(const std::string &command);
    ~StdioTransport();

    StdioTransport(const StdioTransport &) = delete;
    StdioTransport &operator=(const StdioTransport &) = delete;

    json send_request(const std::string &method,
                      const json &params = json::object()) override;

   private:
    std::string command_;
    FILE *in_ = nullptr;   // write to subprocess stdin
    FILE *out_ = nullptr;  // read from subprocess stdout
};

}  // namespace mcp
}  // namespace serius
