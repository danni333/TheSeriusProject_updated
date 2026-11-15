#pragma once

#include <cstdint>
#include <string>

namespace serius {
namespace defaults {

/// Shared defaults so the CLI, server config and stress tests don't each
/// carry their own copy of the same hardcoded values.
inline constexpr uint16_t kHttpPort = 8080;
inline const std::string kServerUrl = "http://localhost:" + std::to_string(kHttpPort);
/// Empty = let the server pick the primary loaded model (multi-model registry
/// returns 404 for unknown names, so a hardcoded default would break stress runs).
inline const std::string kModelName = "";

}  // namespace defaults
}  // namespace serius
