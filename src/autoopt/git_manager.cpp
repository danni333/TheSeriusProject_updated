#include "git_manager.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <iomanip>

namespace serius {
namespace autoopt {

GitManager::GitManager(std::string repo_path) : repo_path_(std::move(repo_path))
{
}

std::string GitManager::format_commit_message(const std::string &kernel_name,
                                              float old_ms,
                                              float new_ms) const
{
    float speedup_pct = 0.0f;
    if (old_ms > 1e-6f) {
        speedup_pct = (1.0f - new_ms / old_ms) * 100.0f;
    }

    std::ostringstream oss;
    oss << "perf: " << kernel_name << " "
        << std::fixed << std::setprecision(1) << old_ms << "ms -> "
        << new_ms << "ms ("
        << std::setprecision(0) << speedup_pct << "% faster)";

    return oss.str();
}

bool GitManager::commit_optimization(const std::string &kernel_name,
                                     float old_ms,
                                     float new_ms,
                                     bool dry_run)
{
    std::string msg = format_commit_message(kernel_name, old_ms, new_ms);
    commit_history_.push_back(msg);

    if (dry_run) {
        return true;
    }

    std::string cmd = "git -C \"" + repo_path_ + "\" commit -am \"" + msg + "\"";
    int ret = std::system(cmd.c_str());
    return (ret == 0);
}

}  // namespace autoopt
}  // namespace serius
