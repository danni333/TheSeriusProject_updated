#pragma once

#include <cstdio>
#include <string>

#include "transport.h"

namespace serius {
namespace mcp {

/// stdio transport: launches a subprocess and communicates over stdin/stdout
/// using newline-delimited JSON-RPC. Requests block until the matching
/// response id arrives or the subprocess closes stdout.
class StdioTransport : public Transport
{
   public:
    explicit StdioTransport(const std::string &command);
    ~StdioTransport();

    StdioTransport(const StdioTransport &) = delete;
    StdioTransport &operator=(const StdioTransport &) = delete;

    json send_request(const std::string &method, const json &params = json::object()) override;

   private:
    void spawn();

    std::string command_;
    FILE *in_ = nullptr;   // write to subprocess stdin
    FILE *out_ = nullptr;  // read from subprocess stdout
#if defined(_WIN32)
    void *process_ = nullptr;  // HANDLE, kept opaque to avoid windows.h here
#else
    long pid_ = -1;
#endif
};

}  // namespace mcp
}  // namespace serius
