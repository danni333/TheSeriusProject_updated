#include "cli/cmd_chat.h"
#include "cli/args_parser.h"

#include <iostream>

namespace serius {
namespace cli {

int cmd_chat(int argc, char **argv)
{
    ArgsParser parser("serius chat", "Interactive chat with the model");

    parser.add_flag({"model", "m", "Path to model file", true, "./models/model.gguf"});
    parser.add_flag({"threads", "t", "Number of threads", true, "4"});
    parser.add_flag({"temperature", "", "Sampling temperature", true, "0.7"});
    parser.add_flag({"max-tokens", "", "Maximum tokens to generate", true, "256"});

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
    int threads = parser.get_int("threads", 4);
    float temperature = parser.get_float("temperature", 0.7f);
    int max_tokens = parser.get_int("max-tokens", 256);

    std::cout << "Loading model: " << model_path << "\n";
    std::cout << "Threads: " << threads << "\n";
    std::cout << "Temperature: " << temperature << "\n";
    std::cout << "Max tokens: " << max_tokens << "\n";
    std::cout << "\n[Chat mode - implementation pending]\n";
    std::cout << "This command will load the model and start an interactive chat session.\n";

    return 0;
}

}  // namespace cli
}  // namespace serius
