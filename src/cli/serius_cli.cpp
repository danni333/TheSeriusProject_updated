#include "cli/cli.h"
#include "cli/cmd_chat.h"
#include "cli/cmd_serve.h"
#include "cli/cmd_doctor.h"
#include "cli/cmd_import.h"
#include "cli/cmd_bench.h"
#include "cli/cmd_stress.h"

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <vector>
#include <string>
#endif

namespace serius {
namespace cli {

int main_cli(int argc, char **argv)
{
    CLI cli;

    cli.register_command({"chat", "Interactive chat with the model", cmd_chat});
    cli.register_command({"serve", "Start HTTP API server", cmd_serve});
    cli.register_command({"doctor", "System diagnostics", cmd_doctor});
    cli.register_command({"import", "Import model from HuggingFace Hub or URL", cmd_import});
    cli.register_command({"bench", "Run performance benchmarks", cmd_bench});
    cli.register_command({"stress", "Run stress tests on the server", cmd_stress});

    return cli.run(argc, argv);
}

}  // namespace cli
}  // namespace serius

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    int w_argc = 0;
    LPWSTR *w_argv = CommandLineToArgvW(GetCommandLineW(), &w_argc);
    if (w_argv) {
        std::vector<std::string> utf8_args(w_argc);
        std::vector<char*> utf8_argv(w_argc);
        for (int i = 0; i < w_argc; ++i) {
            int len = WideCharToMultiByte(CP_UTF8, 0, w_argv[i], -1, nullptr, 0, nullptr, nullptr);
            if (len > 0) {
                utf8_args[i].resize(len - 1);
                WideCharToMultiByte(CP_UTF8, 0, w_argv[i], -1, &utf8_args[i][0], len, nullptr, nullptr);
            }
            utf8_argv[i] = &utf8_args[i][0];
        }
        LocalFree(w_argv);
        return serius::cli::main_cli(w_argc, utf8_argv.data());
    }
#endif
    return serius::cli::main_cli(argc, argv);
}

