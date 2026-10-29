#include "cli/cli.h"
#include "cli/cmd_chat.h"
#include "cli/cmd_serve.h"
#include "cli/cmd_doctor.h"
#include "cli/cmd_import.h"
#include "cli/cmd_bench.h"
#include "cli/cmd_stress.h"

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
    return serius::cli::main_cli(argc, argv);
}
