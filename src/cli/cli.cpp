#include "cli.h"

#include <iostream>

namespace serius {
namespace cli {

CLI::CLI()
{
}

void CLI::register_command(const Command &cmd)
{
    commands_.push_back(cmd);
}

const Command *CLI::find_command(const std::string &name) const
{
    for (const auto &cmd : commands_) {
        if (cmd.name == name) {
            return &cmd;
        }
    }
    return nullptr;
}

int CLI::run(int argc, char **argv)
{
    if (argc < 2) {
        print_help();
        return 0;
    }

    std::string arg = argv[1];

    // Handle global flags
    if (arg == "--help" || arg == "-h") {
        print_help();
        return 0;
    }

    if (arg == "--version" || arg == "-v") {
        print_version();
        return 0;
    }

    // Find and execute command
    const Command *cmd = find_command(arg);
    if (!cmd) {
        std::cerr << "Unknown command: " << arg << "\n\n";
        print_help();
        return 1;
    }

    // Shift arguments (remove command name)
    return cmd->handler(argc - 1, argv + 1);
}

void CLI::print_help() const
{
    std::cout << "Serius LLM Engine - Command Line Interface\n\n";
    std::cout << "Usage: serius <command> [OPTIONS]\n\n";
    std::cout << "Commands:\n";

    for (const auto &cmd : commands_) {
        std::cout << "  " << cmd.name;
        std::cout << " - " << cmd.description << "\n";
    }

    std::cout << "\nGlobal Options:\n";
    std::cout << "  --help, -h     Show this help message\n";
    std::cout << "  --version, -v  Show version information\n";
}

void CLI::print_version() const
{
    std::cout << "Serius LLM Engine v" << version_ << "\n";
}

}  // namespace cli
}  // namespace serius
