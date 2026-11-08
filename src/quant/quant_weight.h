#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
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

    /// Optional device copy of `bytes` for CUDA decode kernels. Owned by the
    /// process; populated by the chat CLI when GPU inference is enabled.
    void *gpu_data = nullptr;

    /// GPU-friendly repacked layout produced at upload time. Each output row
    /// is padded to a multiple of 16 bytes: first all 32-weight blocks' quant
    /// payloads (block-major, so warp lanes read coalesced 16B chunks), then
    /// the fp16 d scales, then fp16 m scales (Q4_1 only). `dev_bytes` replaces
    /// `bytes` on the device; the fields below describe it.
    std::vector<uint8_t> dev_bytes;
    size_t dev_stride = 0;   ///< bytes per output row in dev_bytes
    size_t dev_qs = 0;       ///< quant payload bytes per 32-weight block (16 or 32)
    size_t dev_d = 0;        ///< row offset of the fp16 d scale array
    size_t dev_m = 0;        ///< row offset of the fp16 m scale array (0 = none)
    int dev_kind = 0;        ///< 3=Q4_0, 4=Q4_1, 5=Q8_0

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

/// Repack packed GGUF rows into the aligned `dev_bytes` layout described in
/// QuantWeight. Layout per output row (row stride = dev_stride, multiple of 16):
///   [bpr * qs  quant payload bytes, block-major][bpr * 2B fp16 d scales]
///   [bpr * 2B fp16 m scales, Q4_1 only][pad to 16]
/// Quant chunks are contiguous so a warp reading block `lane` gets fully
/// coalesced 16B loads; scales are likewise contiguous fp16 arrays.
/// No-op for unsupported types or non-32-multiple `in`.
inline void repack_quant_for_gpu(QuantWeight &w)
{
    if (!w.valid() || (w.in % 32) != 0 || w.dev_stride != 0) { return; }
    const size_t bpr = w.blocks_per_row();
    const size_t src_bb = QuantWeight::block_bytes(w.type);
    size_t qs, d_sz = 2, m_sz = 0;
    switch (w.type) {
        case QuantType::Q4_0: qs = 16; break;
        case QuantType::Q4_1: qs = 16; m_sz = 2; break;
        case QuantType::Q8_0: qs = 32; break;
        default: return;
    }
    const size_t d_off = bpr * qs;
    const size_t m_off = (m_sz != 0) ? d_off + bpr * d_sz : 0;
    size_t raw = d_off + bpr * d_sz + (m_sz != 0 ? bpr * m_sz : 0);
    const size_t stride = (raw + 15) & ~size_t(15);
    const size_t out_rows = w.out;
    w.dev_bytes.assign(out_rows * stride, 0);
    for (size_t r = 0; r < out_rows; ++r) {
        const uint8_t *src = w.bytes.data() + r * bpr * src_bb;
        uint8_t *dst = w.dev_bytes.data() + r * stride;
        uint8_t *qd = dst;
        uint8_t *dd = dst + d_off;
        uint8_t *mm = dst + m_off;
        for (size_t b = 0; b < bpr; ++b) {
            const uint8_t *s = src + b * src_bb;
            const size_t qs_src = (w.type == QuantType::Q4_1) ? 4 : 2;
            std::memcpy(qd + b * qs, s + qs_src, qs);  // quant payload, block-major
            std::memcpy(dd + b * 2, s, 2);             // fp16 d
            if (m_sz != 0) { std::memcpy(mm + b * 2, s + 2, 2); }  // fp16 m
        }
    }
    w.dev_stride = stride;
    w.dev_qs = qs;
    w.dev_d = d_off;
    w.dev_m = m_off;
    w.dev_kind = static_cast<int>(w.type);
}

}  // namespace quant
}  // namespace serius
