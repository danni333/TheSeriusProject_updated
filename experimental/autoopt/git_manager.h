#pragma once

#include <string>
#include <vector>

namespace serius {
namespace autoopt {

/// Autonomous Git commit manager for recursive self-improvement records.
class GitManager
{
   public:
    explicit GitManager(std::string repo_path = ".");

    /// Format standardized performance commit message.
    /// Example: "perf: matmul_avx 12.4ms -> 8.1ms (35% faster)"
    std::string format_commit_message(const std::string &kernel_name,
                                      float old_ms,
                                      float new_ms) const;

    /// Commit the optimization to source control.
    /// @param dry_run If true, records the message in internal log without executing system shell git command
    bool commit_optimization(const std::string &kernel_name,
                             float old_ms,
                             float new_ms,
                             bool dry_run = true);

    const std::vector<std::string> &commit_history() const { return commit_history_; }
    void clear_history() { commit_history_.clear(); }

    const std::string &repo_path() const { return repo_path_; }

   private:
    std::string repo_path_;
    std::vector<std::string> commit_history_;
};

}  // namespace autoopt
}  // namespace serius
