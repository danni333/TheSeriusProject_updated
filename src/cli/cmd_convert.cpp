#include "cli/cmd_convert.h"

#include <iostream>

#include "cli/args_parser.h"
#include "io/gguf_export.h"
#include "io/gguf_writer.h"

namespace serius {
namespace cli {

namespace {

bool has_suffix(const std::string &s, const std::string &suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

int cmd_convert(int argc, char **argv)
{
    ArgsParser parser("serius convert", "Convert between GGUF and HuggingFace safetensors formats");

    parser.add_flag({"input", "i", "Input .gguf file, .safetensors file or model dir", true, ""});
    parser.add_flag({"output", "o", "Output .gguf file or directory", true, ""});
    parser.add_flag({"f32", "", "Write GGUF tensors as FP32 instead of FP16 (import only)", false, ""});
    parser.add_flag({"no-tokenizer", "", "Do not embed tokenizer metadata in GGUF (import only)", false, ""});

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

    // GGUF -> HF directory
    if (has_suffix(input, ".gguf")) {
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

    // HF dir / .safetensors -> GGUF
    if (output.empty()) {
        output = "model.gguf";
    }
    if (!has_suffix(output, ".gguf")) {
        output += ".gguf";
    }

    io::GGUFWritOptions opt;
    opt.use_f32 = parser.has("f32");
    opt.embed_tokenizer = !parser.has("no-tokenizer");

    std::cout << "Converting " << input << " -> " << output << "\n";
    if (!io::write_hf_dir_to_gguf(input, output, opt)) {
        std::cerr << "Conversion failed\n";
        return 1;
    }
    std::cout << "Done. Use: serius serve -m " << output << "\n";
    return 0;
}

}  // namespace cli
}  // namespace serius
