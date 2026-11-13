#include "cli/cmd_convert.h"
#include "cli/args_parser.h"
#include "io/gguf_export.h"

#include <iostream>

namespace serius {
namespace cli {

int cmd_convert(int argc, char **argv)
{
    ArgsParser parser("serius convert", "Convert a GGUF model to a safetensors directory");

    parser.add_flag({"input", "i", "Input .gguf file", true, ""});
    parser.add_flag({"output", "o", "Output directory", true, ""});

    if (!parser.parse(argc, argv)) {
        std::cerr << "Error: " << parser.error() << "\n";
        parser.print_help();
        return 1;
    }

    if (parser.has("help")) {
        parser.print_help();
        return 0;
    }

    const std::string input = parser.get("input");
    std::string output = parser.get("output");

    if (input.empty()) {
        std::cerr << "Error: --input is required\n";
        return 1;
    }
    if (output.empty()) {
        output = input.substr(0, input.size() - 5) + "-hf";
    }

    std::cout << "Converting " << input << " -> " << output << "\n";
    if (!io::export_gguf_to_hf_dir(input, output)) {
        std::cerr << "Conversion failed\n";
        return 1;
    }
    std::cout << "Done. Use: serius chat --model " << output << "\n";
    return 0;
}

}  // namespace cli
}  // namespace serius
