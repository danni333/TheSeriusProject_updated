#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace serius {
namespace quant {

/// Quantization formats supported by the on-the-fly GEMV decode kernels.
/// Values mirror ggml/GGUF type ids where applicable.
enum class QuantType : uint8_t
{
    UNSUPPORTED = 0,
    F32 = 1,
    F16 = 2,
    Q4_0 = 3,   // ggml type 2: fp16 scale + 32 x 4-bit, w = (q-8)*d
    Q4_1 = 4,   // ggml type 3: fp16 d,m + 32 x 4-bit, w = q*d + m
    Q8_0 = 5,   // ggml type 8: fp16 scale + 32 x int8, w = q*d
};

/// A weight matrix kept in packed quantized form for fast M==1 inference.
///
/// Layout: rows = output features (neurons), each row holds `in` weights
/// stored as contiguous blocks of 32 along the inner dimension — identical
/// to the ggml/GGUF row order, which is exactly what the GEMV kernels need.
struct QuantWeight
{
    QuantType type = QuantType::UNSUPPORTED;
    size_t in = 0;    ///< inner dim K (features per output row)
    size_t out = 0;   ///< outer dim N (number of output rows)
    std::vector<uint8_t> bytes;  ///< packed rows, row-major [out][row_bytes]

    bool valid() const { return type != QuantType::UNSUPPORTED && in > 0 && out > 0 && !bytes.empty(); }

    /// Bytes per 32-weight block for the given type.
    static constexpr size_t block_bytes(QuantType t)
    {
        switch (t) {
            case QuantType::Q4_0: return 18;
            case QuantType::Q4_1: return 20;
            case QuantType::Q8_0: return 34;
            default: return 0;
        }
    }

    /// Number of 32-weight blocks per output row (ceil).
    size_t blocks_per_row() const { return (in + 31) / 32; }

    /// Bytes per output row.
    size_t row_bytes() const { return blocks_per_row() * block_bytes(type); }
};

}  // namespace quant
}  // namespace serius
