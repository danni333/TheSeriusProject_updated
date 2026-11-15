#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace serius {
namespace defaults {

/// Shared defaults so the CLI, server config and stress tests don't each
/// carry their own copy of the same hardcoded values.

// --- Network ---
inline constexpr uint16_t kHttpPort = 8080;
inline constexpr uint16_t kHubPort = 8081;
inline const std::string kHost = "0.0.0.0";
inline const std::string kServerUrl = "http://localhost:" + std::to_string(kHttpPort);

// --- HTTP client timeouts (seconds) ---
inline constexpr int kMcpConnectTimeoutSec = 10;
inline constexpr int kMcpReadTimeoutSec = 60;
inline constexpr int kStressTimeoutSec = 30;
inline constexpr int kStressChaosTimeoutSec = 5;

// --- Generation defaults ---
inline constexpr size_t kMaxNewTokens = 256;
inline constexpr float kTemperature = 0.7f;
inline constexpr int kTopK = 40;
inline constexpr float kTopP = 0.9f;

/// Empty = let the server pick the primary loaded model (multi-model registry
/// returns 404 for unknown names, so a hardcoded default would break stress runs).
inline const std::string kModelName = "";

}  // namespace defaults
}  // namespace serius
