#pragma once

#include <functional>
#include <string>
#include <vector>

namespace serius {
namespace cli {

/// Command handler function signature.
using CommandHandler = std::function<int(int argc, char **argv)>;

/// Command descriptor.
struct Command
{
    std::string name;
    std::string description;
    CommandHandler handler;
};

/// Main CLI dispatcher.
class CLI
{
   public:
    CLI();

    /// Register a command.
    void register_command(const Command &cmd);

    /// Run the CLI with given arguments.
    /// Returns exit code (0 on success, non-zero on error).
    int run(int argc, char **argv);

    /// Print help message.
    void print_help() const;

    /// Print version.
    void print_version() const;

    const std::string &version() const { return version_; }

   private:
    std::vector<Command> commands_;
    std::string version_ = "0.1.0";

    /// Find command by name.
    const Command *find_command(const std::string &name) const;
};

}  // namespace cli
}  // namespace serius
