#include "cli/cmd_serve.h"
#include "cli/args_parser.h"

#include <iostream>

namespace serius {
namespace cli {

int cmd_serve(int argc, char **argv)
{
    ArgsParser parser("serius serve", "Start HTTP API server");

    parser.add_flag({"model", "m", "Path to model file", true, "./models/model.gguf"});
    parser.add_flag({"port", "p", "Server port", true, "8080"});
    parser.add_flag({"host", "", "Server host", true, "0.0.0.0"});
    parser.add_flag({"gpu", "", "Use GPU (auto/cpu/cuda)", true, "auto"});

    if (!parser.parse(argc, argv)) {
        std::cerr << "Error: " << parser.error() << "\n";
        parser.print_help();
        return 1;
    }

    if (parser.has("help")) {
        parser.print_help();
        return 0;
    }

    std::string model_path = parser.get("model");
    int port = parser.get_int("port", 8080);
    std::string host = parser.get("host", "0.0.0.0");
    std::string gpu = parser.get("gpu", "auto");

    std::cout << "Starting server...\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Host: " << host << ":" << port << "\n";
    std::cout << "GPU: " << gpu << "\n";
    std::cout << "\n[Server mode - implementation pending]\n";
    std::cout << "This command will start the HTTP server using the existing Server class.\n";

    return 0;
}

}  // namespace cli
}  // namespace serius
