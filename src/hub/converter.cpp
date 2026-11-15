#include "converter.h"

#include <filesystem>
#include <iostream>

#include "../io/gguf_writer.h"

namespace fs = std::filesystem;

namespace serius {
namespace hub {

ModelConverter::ModelConverter()
{
}

bool ModelConverter::convert(const std::string &input_path,
                            const std::string &output_path,
                            QuantType quant)
{
    if (quant != QuantType::None && quant != QuantType::FP16) {
        std::cerr << "[Converter] warning: quantization '" << static_cast<int>(quant)
                  << "' not supported yet, writing FP16 tensors\n";
    }
    // Determine format based on extension
    if (input_path.find(".safetensors") != std::string::npos ||
        fs::is_directory(fs::path(input_path))) {
        return safetensors_to_gguf(input_path, output_path);
    }

    std::cerr << "Unsupported input format: " << input_path << "\n";
    return false;
}

bool ModelConverter::safetensors_to_gguf(const std::string &input, const std::string &output)
{
    std::cout << "Converting " << input << " to " << output << " (Safetensors -> GGUF)\n";

    // Accept either a model directory or a .safetensors file directly;
    // gguf_writer resolves config.json/tokenizer from the parent dir.
    std::error_code ec;
    fs::path in_path(input);
    if (!fs::exists(in_path, ec)) {
        std::cerr << "[Converter] input not found: " << input << "\n";
        return false;
    }

    io::GGUFWritOptions opt;  // FP16 tensors (ggml F16), tokenizer embedded
    return io::write_hf_dir_to_gguf(input, output, opt);
}

bool ModelConverter::requantize(const std::string &input, const std::string &output, QuantType quant)
{
    std::cout << "Requantizing " << input << " to " << output << "\n";
    std::cout << "[Requantization implementation pending]\n";

    return false;
}

std::vector<std::string> ModelConverter::supported_input_formats() const
{
    return {"safetensors", "gguf", "bin"};
}

std::vector<std::string> ModelConverter::supported_output_formats() const
{
    return {"gguf"};
}

bool ModelConverter::convert_safetensors_to_gguf_impl(const std::string &input, const std::string &output)
{
    // Full implementation would go here
    return false;
}

}  // namespace hub
}  // namespace serius
