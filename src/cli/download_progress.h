#pragma once

#include <cstdio>

#include "../hub/downloader.h"

namespace serius {
namespace cli {

/// ProgressFn that prints download progress to stderr using carriage returns.
/// Shows "X / Y MiB" when the total size is known, "X MiB" otherwise.
inline hub::ProgressFn make_download_progress()
{
    return [](size_t downloaded, size_t total) {
        constexpr double kMiB = 1048576.0;
        if (total > 0) {
            std::fprintf(stderr, "\r[download] %.1f / %.1f MiB", static_cast<double>(downloaded) / kMiB,
                         static_cast<double>(total) / kMiB);
        } else {
            std::fprintf(stderr, "\r[download] %.1f MiB", static_cast<double>(downloaded) / kMiB);
        }
        std::fflush(stderr);
    };
}

/// Terminate the \r-based progress line after a download attempt.
inline void finish_download_progress()
{
    std::fprintf(stderr, "\n");
}

}  // namespace cli
}  // namespace serius
