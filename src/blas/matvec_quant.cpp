#include "blas.h"

#include "../quant/quant_weight.h"

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <immintrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#endif

namespace serius {
namespace core {
namespace blas {

namespace {

// Branchless fp16 -> fp32. Correct for normal values (the only case that
// occurs for quantization scales); subnormals flush to ~0 and inf/nan are
// not produced by ggml quantizers.
inline float fp16_to_fp32_fast(uint16_t h)
{
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    const uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;
    const uint32_t bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t load_fp16(const uint8_t *p)
{
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

// Scalar dequant+dot helpers for row tails (in % 32 != 0).
// Element e of a block: e<16 -> low nibble of qs[e]; e>=16 -> high nibble of qs[e-16].
inline float q40_elem(const uint8_t *blk, size_t e, float d)
{
    const uint8_t b = blk[2 + (e < 16 ? e : e - 16)];
    const uint32_t nib = (e < 16) ? (b & 0x0F) : ((b >> 4) & 0x0F);
    return (static_cast<int>(nib) - 8) * d;
}

inline float q41_elem(const uint8_t *blk, size_t e, float d, float m)
{
    const uint8_t b = blk[4 + (e < 16 ? e : e - 16)];
    const uint32_t nib = (e < 16) ? (b & 0x0F) : ((b >> 4) & 0x0F);
    return static_cast<int>(nib) * d + m;
}

inline float q80_elem(const uint8_t *blk, size_t e, float d)
{
    return static_cast<int>(static_cast<int8_t>(blk[2 + e])) * d;
}

#if defined(__AVX512F__) || defined(_MSC_VER)

// ---------------- AVX-512 kernels ----------------

// Two-row variant: processes rows row0/row1 together, sharing x vector loads
// and doubling outstanding memory streams (helps DRAM-latency-bound decode).
void q40_row2_avx512(const uint8_t *row0, const uint8_t *row1,
                     size_t in, const float *x, float &out0, float &out1)
{
    const size_t full_blocks = in / 32;
    __m512 acc0[8];
    __m512 acc1[8];
    for (int i = 0; i < 8; ++i) { acc0[i] = _mm512_setzero_ps(); acc1[i] = _mm512_setzero_ps(); }
    const __m512i nib_mask = _mm512_set1_epi32(0x0F);
    const __m512i eight = _mm512_set1_epi32(8);

    size_t k = 0;
    size_t b = 0;
    for (; b + 4 <= full_blocks; b += 4, k += 128) {
        for (int i = 0; i < 4; ++i) {
            const __m512 xlo = _mm512_loadu_ps(x + k + 32 * i);
            const __m512 xhi = _mm512_loadu_ps(x + k + 32 * i + 16);
            {
                const uint8_t *blk = row0 + (b + i) * 18;
                const __m512 d = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
                const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
                const __m512i bytes = _mm512_cvtepu8_epi32(q);
                const __m512i lo = _mm512_and_si512(bytes, nib_mask);
                const __m512i hi = _mm512_and_si512(_mm512_srli_epi32(bytes, 4), nib_mask);
                const __m512 wlo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(lo, eight)), d);
                const __m512 whi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(hi, eight)), d);
                acc0[2 * i]     = _mm512_fmadd_ps(wlo, xlo, acc0[2 * i]);
                acc0[2 * i + 1] = _mm512_fmadd_ps(whi, xhi, acc0[2 * i + 1]);
            }
            {
                const uint8_t *blk = row1 + (b + i) * 18;
                const __m512 d = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
                const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
                const __m512i bytes = _mm512_cvtepu8_epi32(q);
                const __m512i lo = _mm512_and_si512(bytes, nib_mask);
                const __m512i hi = _mm512_and_si512(_mm512_srli_epi32(bytes, 4), nib_mask);
                const __m512 wlo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(lo, eight)), d);
                const __m512 whi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(hi, eight)), d);
                acc1[2 * i]     = _mm512_fmadd_ps(wlo, xlo, acc1[2 * i]);
                acc1[2 * i + 1] = _mm512_fmadd_ps(whi, xhi, acc1[2 * i + 1]);
            }
        }
    }
    for (; b < full_blocks; ++b, k += 32) {
        const __m512 xlo = _mm512_loadu_ps(x + k);
        const __m512 xhi = _mm512_loadu_ps(x + k + 16);
        {
            const uint8_t *blk = row0 + b * 18;
            const __m512 d = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
            const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
            const __m512i bytes = _mm512_cvtepu8_epi32(q);
            const __m512i lo = _mm512_and_si512(bytes, nib_mask);
            const __m512i hi = _mm512_and_si512(_mm512_srli_epi32(bytes, 4), nib_mask);
            const __m512 wlo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(lo, eight)), d);
            const __m512 whi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(hi, eight)), d);
            acc0[0] = _mm512_fmadd_ps(wlo, xlo, acc0[0]);
            acc0[1] = _mm512_fmadd_ps(whi, xhi, acc0[1]);
        }
        {
            const uint8_t *blk = row1 + b * 18;
            const __m512 d = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
            const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
            const __m512i bytes = _mm512_cvtepu8_epi32(q);
            const __m512i lo = _mm512_and_si512(bytes, nib_mask);
            const __m512i hi = _mm512_and_si512(_mm512_srli_epi32(bytes, 4), nib_mask);
            const __m512 wlo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(lo, eight)), d);
            const __m512 whi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(hi, eight)), d);
            acc1[0] = _mm512_fmadd_ps(wlo, xlo, acc1[0]);
            acc1[1] = _mm512_fmadd_ps(whi, xhi, acc1[1]);
        }
    }
    auto hsum = [](__m512 *acc) {
        return _mm512_reduce_add_ps(_mm512_add_ps(
            _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3])),
            _mm512_add_ps(_mm512_add_ps(acc[4], acc[5]), _mm512_add_ps(acc[6], acc[7]))));
    };
    out0 = hsum(acc0);
    out1 = hsum(acc1);
    for (size_t e = in - (in % 32); e < in; ++e) {
        const size_t bi = e / 32;
        const uint8_t *b0 = row0 + bi * 18;
        const uint8_t *b1 = row1 + bi * 18;
        out0 += q40_elem(b0, e % 32, fp16_to_fp32_fast(load_fp16(b0))) * x[e];
        out1 += q40_elem(b1, e % 32, fp16_to_fp32_fast(load_fp16(b1))) * x[e];
    }
}

void q40_row_avx512(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t full_blocks = in / 32;
    __m512 acc[16];
    for (int i = 0; i < 16; ++i) { acc[i] = _mm512_setzero_ps(); }
    const __m512i nib_mask = _mm512_set1_epi32(0x0F);
    const __m512i eight = _mm512_set1_epi32(8);

    size_t k = 0;
    size_t b = 0;
    for (; b + 8 <= full_blocks; b += 8, k += 256) {
        _mm_prefetch(reinterpret_cast<const char *>(row + (b + 16) * 18), _MM_HINT_T0);
        for (int i = 0; i < 8; ++i) {
            const uint8_t *blk = row + (b + i) * 18;
            const __m512 d = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
            const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
            const __m512i bytes = _mm512_cvtepu8_epi32(q);
            const __m512i lo = _mm512_and_si512(bytes, nib_mask);
            const __m512i hi = _mm512_and_si512(_mm512_srli_epi32(bytes, 4), nib_mask);
            const __m512 wlo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(lo, eight)), d);
            const __m512 whi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(hi, eight)), d);
            acc[2 * i]     = _mm512_fmadd_ps(wlo, _mm512_loadu_ps(x + k + 32 * i),      acc[2 * i]);
            acc[2 * i + 1] = _mm512_fmadd_ps(whi, _mm512_loadu_ps(x + k + 32 * i + 16), acc[2 * i + 1]);
        }
    }
    for (; b < full_blocks; ++b, k += 32) {
        const uint8_t *blk = row + b * 18;
        const __m512 d = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
        const __m512i bytes = _mm512_cvtepu8_epi32(q);
        const __m512i lo = _mm512_and_si512(bytes, nib_mask);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi32(bytes, 4), nib_mask);
        const __m512 wlo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(lo, eight)), d);
        const __m512 whi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(hi, eight)), d);
        acc[0] = _mm512_fmadd_ps(wlo, _mm512_loadu_ps(x + k),      acc[0]);
        acc[1] = _mm512_fmadd_ps(whi, _mm512_loadu_ps(x + k + 16), acc[1]);
    }
    __m512 s01 = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
    __m512 s23 = _mm512_add_ps(_mm512_add_ps(acc[4], acc[5]), _mm512_add_ps(acc[6], acc[7]));
    __m512 s45 = _mm512_add_ps(_mm512_add_ps(acc[8], acc[9]), _mm512_add_ps(acc[10], acc[11]));
    __m512 s67 = _mm512_add_ps(_mm512_add_ps(acc[12], acc[13]), _mm512_add_ps(acc[14], acc[15]));
    out = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s01, s23), _mm512_add_ps(s45, s67)));
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 18;
        out += q40_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk))) * x[e];
    }
}

void q41_row_avx512(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t full_blocks = in / 32;
    __m512 acc[16];
    for (int i = 0; i < 16; ++i) { acc[i] = _mm512_setzero_ps(); }
    const __m512i nib_mask = _mm512_set1_epi32(0x0F);

    size_t k = 0;
    size_t b = 0;
    for (; b + 8 <= full_blocks; b += 8, k += 256) {
        _mm_prefetch(reinterpret_cast<const char *>(row + (b + 16) * 20), _MM_HINT_T0);
        for (int i = 0; i < 8; ++i) {
            const uint8_t *blk = row + (b + i) * 20;
            const __m512 d = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
            const __m512 m = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk + 2)));
            const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 4));
            const __m512i bytes = _mm512_cvtepu8_epi32(q);
            const __m512i lo = _mm512_and_si512(bytes, nib_mask);
            const __m512i hi = _mm512_and_si512(_mm512_srli_epi32(bytes, 4), nib_mask);
            const __m512 wlo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), d, m);
            const __m512 whi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi), d, m);
            acc[2 * i]     = _mm512_fmadd_ps(wlo, _mm512_loadu_ps(x + k + 32 * i),      acc[2 * i]);
            acc[2 * i + 1] = _mm512_fmadd_ps(whi, _mm512_loadu_ps(x + k + 32 * i + 16), acc[2 * i + 1]);
        }
    }
    for (; b < full_blocks; ++b, k += 32) {
        const uint8_t *blk = row + b * 20;
        const __m512 d = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
        const __m512 m = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk + 2)));
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 4));
        const __m512i bytes = _mm512_cvtepu8_epi32(q);
        const __m512i lo = _mm512_and_si512(bytes, nib_mask);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi32(bytes, 4), nib_mask);
        const __m512 wlo = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lo), d, m);
        const __m512 whi = _mm512_fmadd_ps(_mm512_cvtepi32_ps(hi), d, m);
        acc[0] = _mm512_fmadd_ps(wlo, _mm512_loadu_ps(x + k),      acc[0]);
        acc[1] = _mm512_fmadd_ps(whi, _mm512_loadu_ps(x + k + 16), acc[1]);
    }
    __m512 s01 = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
    __m512 s23 = _mm512_add_ps(_mm512_add_ps(acc[4], acc[5]), _mm512_add_ps(acc[6], acc[7]));
    __m512 s45 = _mm512_add_ps(_mm512_add_ps(acc[8], acc[9]), _mm512_add_ps(acc[10], acc[11]));
    __m512 s67 = _mm512_add_ps(_mm512_add_ps(acc[12], acc[13]), _mm512_add_ps(acc[14], acc[15]));
    out = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s01, s23), _mm512_add_ps(s45, s67)));
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 20;
        out += q41_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk)), fp16_to_fp32_fast(load_fp16(blk + 2))) * x[e];
    }
}

void q80_row_avx512(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t full_blocks = in / 32;
    __m512 acc[16];
    for (int i = 0; i < 16; ++i) { acc[i] = _mm512_setzero_ps(); }

    size_t k = 0;
    size_t b = 0;
    for (; b + 8 <= full_blocks; b += 8, k += 256) {
        _mm_prefetch(reinterpret_cast<const char *>(row + (b + 16) * 34), _MM_HINT_T0);
        for (int i = 0; i < 8; ++i) {
            const uint8_t *blk = row + (b + i) * 34;
            const __m512 d = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
            const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
            const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 18));
            const __m512 w0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q0)), d);
            const __m512 w1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1)), d);
            acc[2 * i]     = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x + k + 32 * i),      acc[2 * i]);
            acc[2 * i + 1] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x + k + 32 * i + 16), acc[2 * i + 1]);
        }
    }
    for (; b < full_blocks; ++b, k += 32) {
        const uint8_t *blk = row + b * 34;
        const __m512 d = _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
        const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
        const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 18));
        const __m512 w0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q0)), d);
        const __m512 w1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1)), d);
        acc[0] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x + k),      acc[0]);
        acc[1] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x + k + 16), acc[1]);
    }
    __m512 s01 = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
    __m512 s23 = _mm512_add_ps(_mm512_add_ps(acc[4], acc[5]), _mm512_add_ps(acc[6], acc[7]));
    __m512 s45 = _mm512_add_ps(_mm512_add_ps(acc[8], acc[9]), _mm512_add_ps(acc[10], acc[11]));
    __m512 s67 = _mm512_add_ps(_mm512_add_ps(acc[12], acc[13]), _mm512_add_ps(acc[14], acc[15]));
    out = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s01, s23), _mm512_add_ps(s45, s67)));
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 34;
        out += q80_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk))) * x[e];
    }
}

#endif  // AVX-512

#if defined(__AVX2__) || defined(_MSC_VER)

// ---------------- AVX2 kernels ----------------

void q40_row_avx2(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t full_blocks = in / 32;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const __m256i nib_lo = _mm256_set1_epi16(0x0F);
    const __m256i eight = _mm256_set1_epi32(8);

    size_t k = 0;
    for (size_t b = 0; b < full_blocks; ++b, k += 32) {
        const uint8_t *blk = row + b * 18;
        const __m256 d = _mm256_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
        const __m256i bytes = _mm256_cvtepu8_epi16(q);          // 16 x u16
        const __m256i lo16 = _mm256_and_si256(bytes, nib_lo);   // weights 0..15
        const __m256i hi16 = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nib_lo);  // 16..31

        const __m256i l0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(lo16));
        const __m256i l1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(lo16, 1));
        const __m256i h0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(hi16));
        const __m256i h1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(hi16, 1));

        const __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(l0, eight)), d);
        const __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(l1, eight)), d);
        const __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(h0, eight)), d);
        const __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(h1, eight)), d);

        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k + 0),  acc0);
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8),  acc1);
        acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + k + 16), acc2);
        acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + k + 24), acc3);
    }
    alignas(32) float buf[32];
    _mm256_store_ps(buf + 0,  acc0);
    _mm256_store_ps(buf + 8,  acc1);
    _mm256_store_ps(buf + 16, acc2);
    _mm256_store_ps(buf + 24, acc3);
    out = 0.0f;
    for (int i = 0; i < 32; ++i) { out += buf[i]; }
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 18;
        out += q40_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk))) * x[e];
    }
}

void q41_row_avx2(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t full_blocks = in / 32;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const __m256i nib_lo = _mm256_set1_epi16(0x0F);

    size_t k = 0;
    for (size_t b = 0; b < full_blocks; ++b, k += 32) {
        const uint8_t *blk = row + b * 20;
        const __m256 d = _mm256_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
        const __m256 m = _mm256_set1_ps(fp16_to_fp32_fast(load_fp16(blk + 2)));
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 4));
        const __m256i bytes = _mm256_cvtepu8_epi16(q);
        const __m256i lo16 = _mm256_and_si256(bytes, nib_lo);
        const __m256i hi16 = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nib_lo);

        const __m256i l0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(lo16));
        const __m256i l1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(lo16, 1));
        const __m256i h0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(hi16));
        const __m256i h1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(hi16, 1));

        const __m256 w0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(l0), d, m);
        const __m256 w1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(l1), d, m);
        const __m256 w2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(h0), d, m);
        const __m256 w3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(h1), d, m);

        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k + 0),  acc0);
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8),  acc1);
        acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + k + 16), acc2);
        acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + k + 24), acc3);
    }
    alignas(32) float buf[32];
    _mm256_store_ps(buf + 0,  acc0);
    _mm256_store_ps(buf + 8,  acc1);
    _mm256_store_ps(buf + 16, acc2);
    _mm256_store_ps(buf + 24, acc3);
    out = 0.0f;
    for (int i = 0; i < 32; ++i) { out += buf[i]; }
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 20;
        out += q41_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk)), fp16_to_fp32_fast(load_fp16(blk + 2))) * x[e];
    }
}

void q80_row_avx2(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t full_blocks = in / 32;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t k = 0;
    for (size_t b = 0; b < full_blocks; ++b, k += 32) {
        const uint8_t *blk = row + b * 34;
        const __m256 d = _mm256_set1_ps(fp16_to_fp32_fast(load_fp16(blk)));
        const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
        const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 18));
        // cvtepi8_epi32 consumes the low 8 bytes of an xmm -> two calls per 16 bytes
        const __m256i j0 = _mm256_cvtepi8_epi32(q0);
        const __m256i j1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8));
        const __m256i j2 = _mm256_cvtepi8_epi32(q1);
        const __m256i j3 = _mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8));
        const __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(j0), d);
        const __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(j1), d);
        const __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(j2), d);
        const __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(j3), d);
        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k + 0),  acc0);
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8),  acc1);
        acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + k + 16), acc2);
        acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + k + 24), acc3);
    }
    alignas(32) float buf[32];
    _mm256_store_ps(buf + 0,  acc0);
    _mm256_store_ps(buf + 8,  acc1);
    _mm256_store_ps(buf + 16, acc2);
    _mm256_store_ps(buf + 24, acc3);
    out = 0.0f;
    for (int i = 0; i < 32; ++i) { out += buf[i]; }
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 34;
        out += q80_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk))) * x[e];
    }
}

#endif  // AVX2

// ---------------- Integer-dot (Q4_x weights x Q8_1 activation) ----------------
// ggml-style: activation quantized per 32-block to int8; weights stay packed.
// Per block: prod = maddubs(nib_u8[32], xq_i8[32]) -> i16 pairs -> i32 lanes,
// accumulated lane-wise into a float accumulator scaled by (d_w * d_x).
// Q4_0 needs -8*sum(x_q) correction per block; Q4_1 needs +m*sum(x_q) term.

struct XQuant {
    std::vector<int8_t> q;   // in bytes, in%32==0
    std::vector<float> d;    // per-block activation scale
    std::vector<float> c;    // per-block d * sum(x_q)  (for corrections)
};

#if defined(__AVX2__) || defined(_MSC_VER)

inline float hsum256_epi32(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return static_cast<float>(_mm_cvtsi128_si32(s));
}

// Quantize x[0..in) into 32-blocks of int8. in must be a multiple of 32.
void quantize_x_q8_avx2(const float *x, size_t in, XQuant &xq)
{
    const size_t nb = in / 32;
    xq.q.resize(in);
    xq.d.resize(nb);
    xq.c.resize(nb);
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    const __m256i v127 = _mm256_set1_epi32(127);

    for (size_t b = 0; b < nb; ++b) {
        const float *xp = x + b * 32;
        __m256 m0 = _mm256_loadu_ps(xp);
        __m256 m1 = _mm256_loadu_ps(xp + 8);
        __m256 m2 = _mm256_loadu_ps(xp + 16);
        __m256 m3 = _mm256_loadu_ps(xp + 24);
        __m256 a0 = _mm256_andnot_ps(sign_mask, m0);
        __m256 a1 = _mm256_andnot_ps(sign_mask, m1);
        __m256 a2 = _mm256_andnot_ps(sign_mask, m2);
        __m256 a3 = _mm256_andnot_ps(sign_mask, m3);
        __m256 mx = _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
        alignas(32) float mb[8];
        _mm256_store_ps(mb, mx);
        float amax = mb[0];
        for (int i = 1; i < 8; ++i) { amax = std::max(amax, mb[i]); }
        const float d = (amax > 1e-20F) ? (amax / 127.0F) : 1.0F;
        const __m256 id = _mm256_set1_ps(1.0F / d);
        __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(m0, id));
        __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(m1, id));
        __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(m2, id));
        __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(m3, id));
        // sum of quantized values
        const float xs = hsum256_epi32(_mm256_add_epi32(
            _mm256_add_epi32(q0, q1), _mm256_add_epi32(q2, q3)));
        // pack 4x epi32x8 -> 32 x i8, in order
        __m128i p0 = _mm_packs_epi32(_mm256_castsi256_si128(q0), _mm256_extracti128_si256(q0, 1));
        __m128i p1 = _mm_packs_epi32(_mm256_castsi256_si128(q1), _mm256_extracti128_si256(q1, 1));
        __m128i p2 = _mm_packs_epi32(_mm256_castsi256_si128(q2), _mm256_extracti128_si256(q2, 1));
        __m128i p3 = _mm_packs_epi32(_mm256_castsi256_si128(q3), _mm256_extracti128_si256(q3, 1));
        __m128i w0 = _mm_packs_epi16(p0, p1);  // x[0..15]
        __m128i w1 = _mm_packs_epi16(p2, p3);  // x[16..31]
        _mm_storeu_si128(reinterpret_cast<__m128i *>(xq.q.data() + b * 32), w0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(xq.q.data() + b * 32 + 16), w1);
        xq.d[b] = d;
        xq.c[b] = d * xs;
        (void)v127;
    }
}

// Q4_0: w = (nib - 8) * d_w  =>  out = d_w*d_x*sum(nib*xq) - 8*d_w*(d_x*sum(xq))
void q40i_row_avx2(const uint8_t *row, size_t in, const XQuant &xq, float &out)
{
    const size_t nb = in / 32;
    __m256 facc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i nib_lo = _mm256_set1_epi16(0x0F);
    float corr = 0.0F;

    for (size_t b = 0; b < nb; ++b) {
        const uint8_t *blk = row + b * 18;
        const float dw = fp16_to_fp32_fast(load_fp16(blk));
        const __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
        const __m128i lo = _mm_and_si128(qs, _mm_set1_epi8(0x0F));
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), _mm_set1_epi8(0x0F));
        const __m256i w = _mm256_set_m128i(hi, lo);   // [lo|hi] = 32 u8 weights
        const __m256i xqv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq.q.data() + b * 32));
        const __m256i prod = _mm256_madd_epi16(_mm256_maddubs_epi16(w, xqv), ones);
        const __m256 sc = _mm256_set1_ps(dw * xq.d[b]);
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod), sc, facc);
        corr += dw * xq.c[b];
    }
    alignas(32) float buf[8];
    _mm256_store_ps(buf, facc);
    out = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7] - 8.0F * corr;
}

// Q4_1: w = nib*d_w + m_w  =>  out = sum(d_w*d_x*prod) + sum(m_w*(d_x*sum_xq))
void q41i_row_avx2(const uint8_t *row, size_t in, const XQuant &xq, float &out)
{
    const size_t nb = in / 32;
    __m256 facc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    float msum = 0.0F;

    for (size_t b = 0; b < nb; ++b) {
        const uint8_t *blk = row + b * 20;
        const float dw = fp16_to_fp32_fast(load_fp16(blk));
        const float mw = fp16_to_fp32_fast(load_fp16(blk + 2));
        const __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 4));
        const __m128i lo = _mm_and_si128(qs, _mm_set1_epi8(0x0F));
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), _mm_set1_epi8(0x0F));
        const __m256i w = _mm256_set_m128i(hi, lo);
        const __m256i xqv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq.q.data() + b * 32));
        const __m256i prod = _mm256_madd_epi16(_mm256_maddubs_epi16(w, xqv), ones);
        const __m256 sc = _mm256_set1_ps(dw * xq.d[b]);
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod), sc, facc);
        msum += mw * xq.c[b];
    }
    alignas(32) float buf[8];
    _mm256_store_ps(buf, facc);
    out = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7] + msum;
}

#endif  // AVX2

#if defined(__AVX512F__) || defined(_MSC_VER)

// Single-block tail helper (AVX2 width): returns full block contribution.
inline float q40i_blk_tail_avx2(const uint8_t *blk, const XQuant &xq, size_t b)
{
    const float dw = fp16_to_fp32_fast(load_fp16(blk));
    const __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
    const __m128i lo = _mm_and_si128(qs, _mm_set1_epi8(0x0F));
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), _mm_set1_epi8(0x0F));
    const __m256i w = _mm256_set_m128i(hi, lo);
    const __m256i xqv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq.q.data() + b * 32));
    const __m256i prod = _mm256_madd_epi16(_mm256_maddubs_epi16(w, xqv), _mm256_set1_epi16(1));
    const int s = static_cast<int>(hsum256_epi32(prod));
    return dw * xq.d[b] * static_cast<float>(s) - 8.0F * dw * xq.c[b];
}

inline float q41i_blk_tail_avx2(const uint8_t *blk, const XQuant &xq, size_t b)
{
    const float dw = fp16_to_fp32_fast(load_fp16(blk));
    const float mw = fp16_to_fp32_fast(load_fp16(blk + 2));
    const __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 4));
    const __m128i lo = _mm_and_si128(qs, _mm_set1_epi8(0x0F));
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), _mm_set1_epi8(0x0F));
    const __m256i w = _mm256_set_m128i(hi, lo);
    const __m256i xqv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq.q.data() + b * 32));
    const __m256i prod = _mm256_madd_epi16(_mm256_maddubs_epi16(w, xqv), _mm256_set1_epi16(1));
    const int s = static_cast<int>(hsum256_epi32(prod));
    return dw * xq.d[b] * static_cast<float>(s) + mw * xq.c[b];
}

// Q4_0 x Q8: two 32-blocks per zmm iteration.
void q40i_row_avx512(const uint8_t *row, size_t in, const XQuant &xq, float &out)
{
    const size_t nb = in / 32;
    const size_t row_b = 18;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const __m512i ones = _mm512_set1_epi16(1);
    const __m128i m0F = _mm_set1_epi8(0x0F);
    float corr = 0.0F;
    size_t b = 0;

    for (; b + 1 < nb; b += 2) {
        const uint8_t *r0 = row + b * row_b;
        const uint8_t *r1 = row + (b + 1) * row_b;
        const float dw0 = fp16_to_fp32_fast(load_fp16(r0));
        const float dw1 = fp16_to_fp32_fast(load_fp16(r1));
        const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r0 + 2));
        const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r1 + 2));
        const __m128i lo0 = _mm_and_si128(q0, m0F);
        const __m128i hi0 = _mm_and_si128(_mm_srli_epi16(q0, 4), m0F);
        const __m128i lo1 = _mm_and_si128(q1, m0F);
        const __m128i hi1 = _mm_and_si128(_mm_srli_epi16(q1, 4), m0F);
        __m512i w = _mm512_castsi128_si512(lo0);
        w = _mm512_inserti32x4(w, hi0, 1);
        w = _mm512_inserti32x4(w, lo1, 2);
        w = _mm512_inserti32x4(w, hi1, 3);
        const __m512i xqv = _mm512_loadu_si512(reinterpret_cast<const void *>(xq.q.data() + b * 32));
        const __m512i prod = _mm512_maddubs_epi16(w, xqv);
        const __m512i p32 = _mm512_madd_epi16(prod, ones);
        const __m256 plo = _mm256_cvtepi32_ps(_mm512_castsi512_si256(p32));
        const __m256 phi = _mm256_cvtepi32_ps(_mm512_extracti32x8_epi32(p32, 1));
        acc0 = _mm256_fmadd_ps(plo, _mm256_set1_ps(dw0 * xq.d[b]), acc0);
        acc1 = _mm256_fmadd_ps(phi, _mm256_set1_ps(dw1 * xq.d[b + 1]), acc1);
        corr += dw0 * xq.c[b] + dw1 * xq.c[b + 1];
    }
    alignas(32) float buf[16];
    _mm256_store_ps(buf, acc0);
    _mm256_store_ps(buf + 8, acc1);
    float sum = 0.0F;
    for (int i = 0; i < 16; ++i) { sum += buf[i]; }
    if (b < nb) {
        sum += q40i_blk_tail_avx2(row + b * row_b, xq, b);
    }
    out = sum - 8.0F * corr;
}

// Q4_1 x Q8: two 32-blocks per zmm iteration.
void q41i_row_avx512(const uint8_t *row, size_t in, const XQuant &xq, float &out)
{
    const size_t nb = in / 32;
    const size_t row_b = 20;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const __m512i ones = _mm512_set1_epi16(1);
    const __m128i m0F = _mm_set1_epi8(0x0F);
    float msum = 0.0F;
    size_t b = 0;

    for (; b + 1 < nb; b += 2) {
        const uint8_t *r0 = row + b * row_b;
        const uint8_t *r1 = row + (b + 1) * row_b;
        const float dw0 = fp16_to_fp32_fast(load_fp16(r0));
        const float dw1 = fp16_to_fp32_fast(load_fp16(r1));
        const float mw0 = fp16_to_fp32_fast(load_fp16(r0 + 2));
        const float mw1 = fp16_to_fp32_fast(load_fp16(r1 + 2));
        const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r0 + 4));
        const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(r1 + 4));
        const __m128i lo0 = _mm_and_si128(q0, m0F);
        const __m128i hi0 = _mm_and_si128(_mm_srli_epi16(q0, 4), m0F);
        const __m128i lo1 = _mm_and_si128(q1, m0F);
        const __m128i hi1 = _mm_and_si128(_mm_srli_epi16(q1, 4), m0F);
        __m512i w = _mm512_castsi128_si512(lo0);
        w = _mm512_inserti32x4(w, hi0, 1);
        w = _mm512_inserti32x4(w, lo1, 2);
        w = _mm512_inserti32x4(w, hi1, 3);
        const __m512i xqv = _mm512_loadu_si512(reinterpret_cast<const void *>(xq.q.data() + b * 32));
        const __m512i prod = _mm512_maddubs_epi16(w, xqv);
        const __m512i p32 = _mm512_madd_epi16(prod, ones);
        const __m256 plo = _mm256_cvtepi32_ps(_mm512_castsi512_si256(p32));
        const __m256 phi = _mm256_cvtepi32_ps(_mm512_extracti32x8_epi32(p32, 1));
        acc0 = _mm256_fmadd_ps(plo, _mm256_set1_ps(dw0 * xq.d[b]), acc0);
        acc1 = _mm256_fmadd_ps(phi, _mm256_set1_ps(dw1 * xq.d[b + 1]), acc1);
        msum += mw0 * xq.c[b] + mw1 * xq.c[b + 1];
    }
    alignas(32) float buf[16];
    _mm256_store_ps(buf, acc0);
    _mm256_store_ps(buf + 8, acc1);
    float sum = 0.0F;
    for (int i = 0; i < 16; ++i) { sum += buf[i]; }
    if (b < nb) {
        sum += q41i_blk_tail_avx2(row + b * row_b, xq, b);
    }
    out = sum + msum;
}

#endif  // AVX-512

// ---------------- Scalar kernels ----------------

void q40_row_scalar(const uint8_t *row, size_t in, const float *x, float &out)
{
    out = 0.0f;
    for (size_t b = 0; b < in / 32; ++b) {
        const uint8_t *blk = row + b * 18;
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        for (size_t e = 0; e < 32; ++e) {
            out += q40_elem(blk, e, d) * x[b * 32 + e];
        }
    }
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 18;
        out += q40_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk))) * x[e];
    }
}

void q41_row_scalar(const uint8_t *row, size_t in, const float *x, float &out)
{
    out = 0.0f;
    for (size_t b = 0; b < in / 32; ++b) {
        const uint8_t *blk = row + b * 20;
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        const float m = fp16_to_fp32_fast(load_fp16(blk + 2));
        for (size_t e = 0; e < 32; ++e) {
            out += q41_elem(blk, e, d, m) * x[b * 32 + e];
        }
    }
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 20;
        out += q41_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk)), fp16_to_fp32_fast(load_fp16(blk + 2))) * x[e];
    }
}

// ggml block_q4_K scale/min extraction (6-bit fields packed across scales[12]).
void q4k_scale_min(const uint8_t *sc, int j, int &d6, int &m6)
{
    if (j < 4) {
        d6 = sc[j] & 63;
        m6 = sc[j + 4] & 63;
    } else {
        d6 = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        m6 = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

// Scalar fallback for native Q4_K rows (raw 144B superblocks, in%256==0).
void q4k_row_scalar(const uint8_t *row, size_t in, const float *x, float &out)
{
    out = 0.0f;
    const size_t nsb = in / 256;
    for (size_t s = 0; s < nsb; ++s) {
        const uint8_t *blk = row + s * 144;
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        const float dmin = fp16_to_fp32_fast(load_fp16(blk + 2));
        const uint8_t *sc = blk + 4;
        const uint8_t *qs = blk + 16;
        for (int sb = 0; sb < 8; ++sb) {
            int d6, m6;
            q4k_scale_min(sc, sb, d6, m6);
            const float D = d * static_cast<float>(d6);
            const float M = dmin * static_cast<float>(m6);
            const uint8_t *qb = qs + (sb >> 1) * 32;
            const bool hi = (sb & 1) != 0;
            const size_t base = (s * 8 + sb) * 32;
            for (int l = 0; l < 32; ++l) {
                const int nib = hi ? (qb[l] >> 4) : (qb[l] & 0xF);
                out += (D * static_cast<float>(nib) - M) * x[base + l];
            }
        }
    }
}

void q80_row_scalar(const uint8_t *row, size_t in, const float *x, float &out)
{
    out = 0.0f;
    for (size_t b = 0; b < in / 32; ++b) {
        const uint8_t *blk = row + b * 34;
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        for (size_t e = 0; e < 32; ++e) {
            out += q80_elem(blk, e, d) * x[b * 32 + e];
        }
    }
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 34;
        out += q80_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk))) * x[e];
    }
}

using RowFn = void (*)(const uint8_t *, size_t, const float *, float &);

RowFn pick_row_fn(quant::QuantType type)
{
    const auto info = get_cpu_info();
#if defined(__AVX512F__) || defined(_MSC_VER)
    static const bool no_avx512 = [] {
#if defined(_MSC_VER)
        char buf[8];
        size_t len = 0;
        return getenv_s(&len, buf, sizeof(buf), "SERIUS_NO_AVX512") == 0 && len > 0;
#else
        return std::getenv("SERIUS_NO_AVX512") != nullptr;
#endif
    }();
    if (info.has_avx512f && !no_avx512) {
        switch (type) {
            case quant::QuantType::Q4_0: return q40_row_avx512;
            case quant::QuantType::Q4_1: return q41_row_avx512;
            case quant::QuantType::Q8_0: return q80_row_avx512;
            default: break;
        }
    }
#endif
#if defined(__AVX2__) || defined(_MSC_VER)
    if (info.has_avx2) {
        switch (type) {
            case quant::QuantType::Q4_0: return q40_row_avx2;
            case quant::QuantType::Q4_1: return q41_row_avx2;
            case quant::QuantType::Q8_0: return q80_row_avx2;
            default: break;
        }
    }
#endif
    switch (type) {
        case quant::QuantType::Q4_0: return q40_row_scalar;
        case quant::QuantType::Q4_1: return q41_row_scalar;
        case quant::QuantType::Q8_0: return q80_row_scalar;
        case quant::QuantType::Q4_K: return q4k_row_scalar;
        default: return nullptr;
    }
}

}  // namespace

/// y[j] = dot(dequant(W_row_j), x[:]) for all N output rows.
/// W is packed ggml-style rows [N][row_bytes]; x has `in` floats; y has `out` floats.
void matvec_q(const quant::QuantWeight &w, const float *x, float *y)
{
    if (!w.valid() || x == nullptr || y == nullptr) { return; }
    const size_t K = w.in;
    const size_t N = w.out;

    // Integer-dot path (ggml-style): Q4_0/Q4_1 weights x int8-quantized
    // activation. Fewer uops/byte than the fp dequant path.
#if defined(__AVX2__) || defined(_MSC_VER)
    // Integer-dot path (ggml-style) is opt-in: on this CPU it measured
    // parity with the fp dequant kernels (DRAM-bound), with slightly worse
    // precision. Enable with SERIUS_INTDOT=1 to experiment.
    static const bool yes_intdot = [] {
#if defined(_MSC_VER)
        char buf[8];
        size_t len = 0;
        return getenv_s(&len, buf, sizeof(buf), "SERIUS_INTDOT") == 0 && len > 0;
#else
        return std::getenv("SERIUS_INTDOT") != nullptr;
#endif
    }();
    if (yes_intdot &&
        (w.type == quant::QuantType::Q4_0 || w.type == quant::QuantType::Q4_1) &&
        (K % 32) == 0 && K >= 32) {
        const auto info = get_cpu_info();
        using IRowFn = void (*)(const uint8_t *, size_t, const XQuant &, float &);
        IRowFn ifn = nullptr;
#if defined(__AVX512F__) || defined(_MSC_VER)
        static const bool no_avx512 = [] {
#if defined(_MSC_VER)
            char buf[8];
            size_t len = 0;
            return getenv_s(&len, buf, sizeof(buf), "SERIUS_NO_AVX512") == 0 && len > 0;
#else
            return std::getenv("SERIUS_NO_AVX512") != nullptr;
#endif
        }();
        if (info.has_avx512f && !no_avx512) {
            ifn = (w.type == quant::QuantType::Q4_0) ? q40i_row_avx512 : q41i_row_avx512;
        }
#endif
        if (ifn == nullptr) {
            if (!info.has_avx2) { goto fp_path; }
            ifn = (w.type == quant::QuantType::Q4_0) ? q40i_row_avx2 : q41i_row_avx2;
        }
        {
            XQuant xq;  // shared read-only across worker threads
            quantize_x_q8_avx2(x, K, xq);
            const size_t row_bytes = w.row_bytes();
            const uint8_t *base = w.bytes.data();
            #pragma omp parallel for schedule(static) if (N >= 256)
            for (int64_t j = 0; j < static_cast<int64_t>(N); ++j) {
                const size_t r = static_cast<size_t>(j);
                float sum = 0.0F;
                ifn(base + r * row_bytes, K, xq, sum);
                y[r] = sum;
            }
        }
        return;
    }
#endif

fp_path:
    RowFn fn = pick_row_fn(w.type);
    if (fn == nullptr) { return; }

    const size_t row_bytes = w.row_bytes();
    const uint8_t *base = w.bytes.data();

    // Q4_0 + AVX-512: two rows per iteration for more memory-level parallelism.
#if defined(__AVX512F__) || defined(_MSC_VER)
    if (fn == q40_row_avx512) {
        const int64_t pairs = static_cast<int64_t>(N / 2);
        #pragma omp parallel for schedule(static) if (pairs >= 128)
        for (int64_t p = 0; p < pairs; ++p) {
            const size_t r0 = static_cast<size_t>(2 * p);
            const size_t r1 = r0 + 1;
            q40_row2_avx512(base + r0 * row_bytes, base + r1 * row_bytes,
                            K, x, y[r0], y[r1]);
        }
        if ((N & 1) != 0) {
            const size_t r = N - 1;
            float sum = 0.0f;
            fn(base + r * row_bytes, K, x, sum);
            y[r] = sum;
        }
        return;
    }
#endif

    #pragma omp parallel for schedule(static) if (N >= 256)
    for (int64_t j = 0; j < static_cast<int64_t>(N); ++j) {
        const size_t r = static_cast<size_t>(j);
        float sum = 0.0f;
        fn(base + r * row_bytes, K, x, sum);
        y[r] = sum;
    }
}

/// Pull `w`'s packed bytes into cache ahead of the GEMV that will consume
/// them. Intended to run during serial sections (attention/norm) so the
/// subsequent matvec_q hits L3 instead of paying full DRAM latency.
void prefetch_quant_weight(const quant::QuantWeight &w)
{
    if (!w.valid()) { return; }
    const uint8_t *p = w.bytes.data();
    const size_t n = w.bytes.size();
    const int64_t lines = static_cast<int64_t>((n + 63) / 64);
    #pragma omp parallel for schedule(static) if (lines >= 256)
    for (int64_t i = 0; i < lines; ++i) {
        _mm_prefetch(reinterpret_cast<const char *>(p) + i * 64, _MM_HINT_T0);
    }
}

}  // namespace blas
}  // namespace core
}  // namespace serius
