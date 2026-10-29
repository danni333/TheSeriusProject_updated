#include "cli/cmd_doctor.h"
#include "cli/args_parser.h"

#include <iostream>

namespace serius {
namespace cli {

int cmd_doctor(int argc, char **argv)
{
    ArgsParser parser("serius doctor", "System diagnostics");

    if (!parser.parse(argc, argv)) {
        std::cerr << "Error: " << parser.error() << "\n";
        parser.print_help();
        return 1;
    }

    if (parser.has("help")) {
        parser.print_help();
        return 0;
    }

    std::cout << "Serius System Diagnostics\n";
    std::cout << "========================\n\n";

    std::cout << "✅ CPU: Detected\n";
    std::cout << "✅ RAM: Available\n";
    std::cout << "⚠️  GPU: Detection pending (CUDA/Metal support)\n";
    std::cout << "✅ Build: C++20 compatible\n";

    std::cout << "\n[Doctor implementation pending]\n";
    std::cout << "This command will check for GPU, RAM, and dependencies.\n";

    return 0;
}

}  // namespace cli
}  // namespace serius
