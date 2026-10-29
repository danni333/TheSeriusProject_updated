#pragma once

#include <string>

namespace serius {
namespace hub {

enum class QuantType
{
    None,
    FP16,
    FP8,
    INT8,
    INT4_AWQ,
    INT4_GPTQ
};

/// Model format converter.
class ModelConverter
{
   public:
    ModelConverter();
    ~ModelConverter() = default;

    /// Convert model between formats.
    /// @param input_path Input model file
    /// @param output_path Output model file
    /// @param quant Quantization type
    /// @return true on success
    bool convert(const std::string &input_path,
                 const std::string &output_path,
                 QuantType quant = QuantType::None);

    /// Convert Safetensors to GGUF.
    bool safetensors_to_gguf(const std::string &input, const std::string &output);

    /// Re-quantize existing model.
    bool requantize(const std::string &input, const std::string &output, QuantType quant);

    /// Get supported input formats.
    std::vector<std::string> supported_input_formats() const;

    /// Get supported output formats.
    std::vector<std::string> supported_output_formats() const;

   private:
    bool convert_safetensors_to_gguf_impl(const std::string &input, const std::string &output);
};

}  // namespace hub
}  // namespace serius
