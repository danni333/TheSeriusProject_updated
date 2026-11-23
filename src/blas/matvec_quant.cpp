#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../quant/quant_weight.h"
#include "blas.h"

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(_M_ARM64)
#include <arm64_neon.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

// x86 SIMD paths require an x86 target; MSVC defines _MSC_VER on ARM64 too,
// where immintrin.h and AVX intrinsics are unavailable.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define SERIUS_SIMD_X86 1
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
    const uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
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

void q4k_scale_min(const uint8_t *sc, int j, int &d6, int &m6);

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

#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))

// ---------------- AVX-512 kernels ----------------

// Two-row variant: processes rows row0/row1 together, sharing x vector loads
// and doubling outstanding memory streams (helps DRAM-latency-bound decode).
void q40_row2_avx512(const uint8_t *row0, const uint8_t *row1, size_t in, const float *x, float &out0, float &out1)
{
    const size_t full_blocks = in / 32;
    __m512 acc0[8];
    __m512 acc1[8];
    for (int i = 0; i < 8; ++i) {
        acc0[i] = _mm512_setzero_ps();
        acc1[i] = _mm512_setzero_ps();
    }
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
                acc0[2 * i] = _mm512_fmadd_ps(wlo, xlo, acc0[2 * i]);
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
                acc1[2 * i] = _mm512_fmadd_ps(wlo, xlo, acc1[2 * i]);
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
        return _mm512_reduce_add_ps(
            _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3])),
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
    for (int i = 0; i < 16; ++i) {
        acc[i] = _mm512_setzero_ps();
    }
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
            acc[2 * i] = _mm512_fmadd_ps(wlo, _mm512_loadu_ps(x + k + 32 * i), acc[2 * i]);
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
        acc[0] = _mm512_fmadd_ps(wlo, _mm512_loadu_ps(x + k), acc[0]);
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
    for (int i = 0; i < 16; ++i) {
        acc[i] = _mm512_setzero_ps();
    }
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
            acc[2 * i] = _mm512_fmadd_ps(wlo, _mm512_loadu_ps(x + k + 32 * i), acc[2 * i]);
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
        acc[0] = _mm512_fmadd_ps(wlo, _mm512_loadu_ps(x + k), acc[0]);
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
    for (int i = 0; i < 16; ++i) {
        acc[i] = _mm512_setzero_ps();
    }

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
            acc[2 * i] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x + k + 32 * i), acc[2 * i]);
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
        acc[0] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x + k), acc[0]);
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

// Q4_K superblock: 256 weights / 144B, 8 sub-blocks of 32.
// w = d*sc*nib - dmin*m. Each 32B qs group feeds two sub-blocks: even sb
// uses low nibbles, odd sb high nibbles. 16 zmm accumulators cover the
// 256 positions of one superblock.
void q4k_row_avx512(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t nsb = in / 256;
    __m512 acc[16];
    for (int i = 0; i < 16; ++i) { acc[i] = _mm512_setzero_ps(); }
    const __m512i nib8 = _mm512_set1_epi8(0x0F);

    size_t k = 0;
    for (size_t s = 0; s < nsb; ++s, k += 256) {
        const uint8_t *blk = row + s * 144;
        _mm_prefetch(reinterpret_cast<const char *>(blk + 144), _MM_HINT_T0);
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        const float dmin = fp16_to_fp32_fast(load_fp16(blk + 2));
        const uint8_t *sc = blk + 4;
        const uint8_t *qs = blk + 16;
        for (int g = 0; g < 4; ++g) {
            // One 32B group -> lo/hi nibble pairs, each expanded to two
            // __m512i of 16 i32 lanes (weights 0..15, 16..31 of the sub-block).
            const __m512i by = _mm512_loadu_si512(reinterpret_cast<const void *>(qs + g * 32));
            const __m512i lo_b = _mm512_and_si512(by, nib8);
            const __m512i hi_b = _mm512_and_si512(_mm512_srli_epi16(by, 4), nib8);
            const __m512i lo0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(lo_b));
            const __m512i lo1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(lo_b, 1));
            const __m512i hi0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(hi_b));
            const __m512i hi1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(hi_b, 1));
            const size_t xo = k + g * 64;  // group covers weights [64g, 64g+64)
            for (int half = 0; half < 2; ++half) {
                // half=0 -> sub-block sb=2g (lo nibbles), half=1 -> sb=2g+1 (hi)
                const int sb = 2 * g + half;
                int d6, m6;
                q4k_scale_min(sc, sb, d6, m6);
                const __m512 D = _mm512_set1_ps(d * static_cast<float>(d6));
                const __m512 M = _mm512_set1_ps(-dmin * static_cast<float>(m6));
                const __m512i n0 = half ? hi0 : lo0;
                const __m512i n1 = half ? hi1 : lo1;
                const size_t xoff = xo + half * 32;
                const __m512 w0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(n0), D, M);
                const __m512 w1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(n1), D, M);
                acc[sb * 2]     = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x + xoff),      acc[sb * 2]);
                acc[sb * 2 + 1] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x + xoff + 16), acc[sb * 2 + 1]);
            }
        }
    }
    __m512 s01 = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
    __m512 s23 = _mm512_add_ps(_mm512_add_ps(acc[4], acc[5]), _mm512_add_ps(acc[6], acc[7]));
    __m512 s45 = _mm512_add_ps(_mm512_add_ps(acc[8], acc[9]), _mm512_add_ps(acc[10], acc[11]));
    __m512 s67 = _mm512_add_ps(_mm512_add_ps(acc[12], acc[13]), _mm512_add_ps(acc[14], acc[15]));
    out = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s01, s23), _mm512_add_ps(s45, s67)));
    for (size_t e = in - (in % 256); e < in; ++e) {
        const size_t s = e / 256;
        const size_t sb = (e % 256) / 32;
        const uint8_t *blk = row + s * 144;
        const uint8_t *sc = blk + 4;
        const uint8_t *qb = blk + 16 + (sb >> 1) * 32;
        const int l = static_cast<int>(e % 32);
        const uint8_t byte = qb[l];
        const int nib = (sb & 1) ? (byte >> 4) : (byte & 0xF);
        int d6, m6;
        q4k_scale_min(sc, static_cast<int>(sb), d6, m6);
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        const float dmin = fp16_to_fp32_fast(load_fp16(blk + 2));
        out += (d * d6 * nib - dmin * m6) * x[e];
    }
}

#endif  // AVX-512

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))

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
        const __m256i bytes = _mm256_cvtepu8_epi16(q);                               // 16 x u16
        const __m256i lo16 = _mm256_and_si256(bytes, nib_lo);                        // weights 0..15
        const __m256i hi16 = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nib_lo);  // 16..31

        const __m256i l0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(lo16));
        const __m256i l1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(lo16, 1));
        const __m256i h0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(hi16));
        const __m256i h1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(hi16, 1));

        const __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(l0, eight)), d);
        const __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(l1, eight)), d);
        const __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(h0, eight)), d);
        const __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(h1, eight)), d);

        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k + 0), acc0);
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8), acc1);
        acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + k + 16), acc2);
        acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + k + 24), acc3);
    }
    alignas(32) float buf[32];
    _mm256_store_ps(buf + 0, acc0);
    _mm256_store_ps(buf + 8, acc1);
    _mm256_store_ps(buf + 16, acc2);
    _mm256_store_ps(buf + 24, acc3);
    out = 0.0f;
    for (int i = 0; i < 32; ++i) {
        out += buf[i];
    }
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

        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k + 0), acc0);
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8), acc1);
        acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + k + 16), acc2);
        acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + k + 24), acc3);
    }
    alignas(32) float buf[32];
    _mm256_store_ps(buf + 0, acc0);
    _mm256_store_ps(buf + 8, acc1);
    _mm256_store_ps(buf + 16, acc2);
    _mm256_store_ps(buf + 24, acc3);
    out = 0.0f;
    for (int i = 0; i < 32; ++i) {
        out += buf[i];
    }
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
        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k + 0), acc0);
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8), acc1);
        acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + k + 16), acc2);
        acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + k + 24), acc3);
    }
    alignas(32) float buf[32];
    _mm256_store_ps(buf + 0, acc0);
    _mm256_store_ps(buf + 8, acc1);
    _mm256_store_ps(buf + 16, acc2);
    _mm256_store_ps(buf + 24, acc3);
    out = 0.0f;
    for (int i = 0; i < 32; ++i) {
        out += buf[i];
    }
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 34;
        out += q80_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk))) * x[e];
    }
}

// Expand 32 packed bytes into four __m256i of 8 i32 lanes each
// (weights 0..31 in order). Byte j maps to output j.
inline void bytes32_to_i32x8(const uint8_t *p, __m256i out[4])
{
    const __m256i b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
    const __m256i b1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 16)));
    out[0] = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b0));
    out[1] = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b0, 1));
    out[2] = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b1));
    out[3] = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b1, 1));
}

// Load 32 bytes, keep the low nibble of each (or the high nibble when
// `high`), and expand to four __m256i of 8 i32 lanes (bytes 0..31 in order).
inline void nibbles32_to_i32x8(const uint8_t *p, bool high, __m256i out[4])
{
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    if (high) {
        v = _mm256_srli_epi16(v, 4);
    }
    v = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
    const __m256i a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v));
    const __m256i b = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1));
    out[0] = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(a));
    out[1] = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(a, 1));
    out[2] = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b));
    out[3] = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b, 1));
}

// Q4_K: 256 weights / 144B superblock, 8 sub-blocks of 32.
// w = d*sc*nib - dmin*m per sub-block; nib from low (even sb) or high (odd sb)
// nibble of qs[(sb/2)*32 + l].
void q4k_row_avx2(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t nsb = in / 256;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t k = 0;
    for (size_t s = 0; s < nsb; ++s, k += 256) {
        const uint8_t *blk = row + s * 144;
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        const float dmin = fp16_to_fp32_fast(load_fp16(blk + 2));
        const uint8_t *sc = blk + 4;
        const uint8_t *qs = blk + 16;
        for (int sb = 0; sb < 8; ++sb) {
            int d6, m6;
            q4k_scale_min(sc, sb, d6, m6);
            const __m256 D = _mm256_set1_ps(d * static_cast<float>(d6));
            const __m256 M = _mm256_set1_ps(-dmin * static_cast<float>(m6));
            const uint8_t *qb = qs + (sb >> 1) * 32;
            __m256i nib[4];
            nibbles32_to_i32x8(qb, (sb & 1) != 0, nib);
            const size_t xo = k + sb * 32;
            for (int i = 0; i < 4; ++i) {
                const __m256 w = _mm256_fmadd_ps(_mm256_cvtepi32_ps(nib[i]), D, M);
                const __m256 xa = (i == 0) ? acc0 : (i == 1) ? acc1 : (i == 2) ? acc2 : acc3;
                const __m256 r = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + xo + 8 * i), xa);
                if (i == 0)
                    acc0 = r;
                else if (i == 1)
                    acc1 = r;
                else if (i == 2)
                    acc2 = r;
                else
                    acc3 = r;
            }
        }
    }
    alignas(32) float buf[32];
    _mm256_store_ps(buf + 0, acc0);
    _mm256_store_ps(buf + 8, acc1);
    _mm256_store_ps(buf + 16, acc2);
    _mm256_store_ps(buf + 24, acc3);
    out = 0.0f;
    for (int i = 0; i < 32; ++i) {
        out += buf[i];
    }
    for (size_t e = in - (in % 256); e < in; ++e) {
        // Tail (only reachable if in % 256 != 0): fall back per element.
        const size_t s = e / 256;
        const size_t sb = (e % 256) / 32;
        const uint8_t *blk = row + s * 144;
        const uint8_t *sc = blk + 4;
        const uint8_t *qb = blk + 16 + (sb >> 1) * 32;
        const int l = static_cast<int>(e % 32);
        const uint8_t byte = qb[l];
        const int nib = (sb & 1) ? (byte >> 4) : (byte & 0xF);
        int d6, m6;
        q4k_scale_min(sc, static_cast<int>(sb), d6, m6);
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        const float dmin = fp16_to_fp32_fast(load_fp16(blk + 2));
        out += (d * d6 * nib - dmin * m6) * x[e];
    }
}

// Q5_K: 256 weights / 176B superblock. Same scale packing as Q4_K;
// q5 = low/high nibble + high bit from qh (bit 2g / 2g+1 of group g).
void q5k_row_avx2(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t nsb = in / 256;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t k = 0;
    for (size_t s = 0; s < nsb; ++s, k += 256) {
        const uint8_t *blk = row + s * 176;
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        const float dmin = fp16_to_fp32_fast(load_fp16(blk + 2));
        const uint8_t *scales = blk + 4;
        const uint8_t *qh = blk + 16;
        const uint8_t *ql = blk + 48;
        // High bits for all 32 qh bytes, expanded once per superblock.
        __m256i hb_all[4];
        bytes32_to_i32x8(qh, hb_all);
        for (int sb = 0; sb < 8; ++sb) {
            const int g = sb >> 1;
            const bool hi = (sb & 1) != 0;
            uint8_t sc8, m8;
            if (sb < 4) {
                sc8 = scales[sb] & 63;
                m8 = scales[sb + 4] & 63;
            } else {
                sc8 = (scales[sb + 4] & 0xF) | ((scales[sb - 4] >> 6) << 4);
                m8 = (scales[sb + 4] >> 4) | ((scales[sb] >> 6) << 4);
            }
            const __m256 D = _mm256_set1_ps(d * static_cast<float>(sc8));
            const __m256 M = _mm256_set1_ps(-dmin * static_cast<float>(m8));
            const uint8_t *qb = ql + g * 32;
            const int bitpos = 2 * g + (hi ? 1 : 0);
            __m256i nib[4];
            nibbles32_to_i32x8(qb, hi, nib);
            // High bit: (qh[l] >> bitpos) & 1, *16, added to nibble value.
            const __m256i bit_mask = _mm256_set1_epi32(1 << bitpos);
            const __m256i sixteen = _mm256_set1_epi32(16);
            for (int i = 0; i < 4; ++i) {
                const __m256i is_set = _mm256_cmpeq_epi32(_mm256_and_si256(hb_all[i], bit_mask), bit_mask);
                nib[i] = _mm256_add_epi32(nib[i], _mm256_and_si256(is_set, sixteen));
            }
            const size_t xo = k + sb * 32;
            for (int i = 0; i < 4; ++i) {
                const __m256 w = _mm256_fmadd_ps(_mm256_cvtepi32_ps(nib[i]), D, M);
                const __m256 xa = (i == 0) ? acc0 : (i == 1) ? acc1 : (i == 2) ? acc2 : acc3;
                const __m256 r = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + xo + 8 * i), xa);
                if (i == 0)
                    acc0 = r;
                else if (i == 1)
                    acc1 = r;
                else if (i == 2)
                    acc2 = r;
                else
                    acc3 = r;
            }
        }
    }
    alignas(32) float buf[32];
    _mm256_store_ps(buf + 0, acc0);
    _mm256_store_ps(buf + 8, acc1);
    _mm256_store_ps(buf + 16, acc2);
    _mm256_store_ps(buf + 24, acc3);
    out = 0.0f;
    for (int i = 0; i < 32; ++i) {
        out += buf[i];
    }
}

#endif  // AVX2

// ---------------- Integer-dot (Q4_x/Q8_0 weights x Q8 activation) ----------------
// ggml-style: activation quantized per 32-block to int8; weights stay packed.
// Per block: prod = maddubs(nib_u8[32], xq_i8[32]) -> i16 pairs -> i32 lanes,
// accumulated lane-wise into a float accumulator scaled by (d_w * d_x).
// Q4_0 needs -8*sum(x_q) correction per block; Q4_1 needs +m*sum(x_q) term.
// Q8_0 weights are already signed i8: q80i_* dot them against an i16 copy of
// the activation (XQuant::q16) via madd_epi16 — exact, no saturation.

struct XQuant
{
    std::vector<int8_t> q;    // in bytes, in%32==0
    std::vector<int16_t> q16; // in int16 lanes (same order) for madd_epi16 kernels
    std::vector<float> d;     // per-block activation scale
    std::vector<float> c;     // per-block d * sum(x_q)  (for corrections)
};

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))

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
    xq.q16.resize(in);
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
        for (int i = 1; i < 8; ++i) {
            amax = std::max(amax, mb[i]);
        }
        const float d = (amax > 1e-20F) ? (amax / 127.0F) : 1.0F;
        const __m256 id = _mm256_set1_ps(1.0F / d);
        __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(m0, id));
        __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(m1, id));
        __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(m2, id));
        __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(m3, id));
        // sum of quantized values
        const float xs = hsum256_epi32(_mm256_add_epi32(_mm256_add_epi32(q0, q1), _mm256_add_epi32(q2, q3)));
        // pack 4x epi32x8 -> 32 x i8, in order
        __m128i p0 = _mm_packs_epi32(_mm256_castsi256_si128(q0), _mm256_extracti128_si256(q0, 1));
        __m128i p1 = _mm_packs_epi32(_mm256_castsi256_si128(q1), _mm256_extracti128_si256(q1, 1));
        __m128i p2 = _mm_packs_epi32(_mm256_castsi256_si128(q2), _mm256_extracti128_si256(q2, 1));
        __m128i p3 = _mm_packs_epi32(_mm256_castsi256_si128(q3), _mm256_extracti128_si256(q3, 1));
        __m128i w0 = _mm_packs_epi16(p0, p1);  // x[0..15]
        __m128i w1 = _mm_packs_epi16(p2, p3);  // x[16..31]
        _mm_storeu_si128(reinterpret_cast<__m128i *>(xq.q.data() + b * 32), w0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(xq.q.data() + b * 32 + 16), w1);
        // i16 copy for madd_epi16-based kernels (Q8_0 weights are signed i8).
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(xq.q16.data() + b * 32),
                            _mm256_cvtepi8_epi16(w0));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(xq.q16.data() + b * 32 + 16),
                            _mm256_cvtepi8_epi16(w1));
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
        const __m256i w = _mm256_set_m128i(hi, lo);  // [lo|hi] = 32 u8 weights
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

// Q8_0 x Q8: weights are already signed int8, so dot them against the i16
// activation copy with madd_epi16 — exact integer math, no saturation issues
// (unlike a maddubs +128 offset trick, which can overflow i16).
// out = sum_b dw_b * dx_b * sum_e q_e * xq_e
void q80i_row_avx2(const uint8_t *row, size_t in, const XQuant &xq, float &out)
{
    const size_t nb = in / 32;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        const uint8_t *blk = row + b * 34;
        const float dw = fp16_to_fp32_fast(load_fp16(blk));
        const __m256i w8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + 2));
        const __m256i wlo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w8));
        const __m256i whi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w8, 1));
        const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq.q16.data() + b * 32));
        const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq.q16.data() + b * 32 + 16));
        const __m256 sc = _mm256_set1_ps(dw * xq.d[b]);
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(wlo, x0)), sc, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(whi, x1)), sc, acc1);
    }
    alignas(32) float buf[16];
    _mm256_store_ps(buf, acc0);
    _mm256_store_ps(buf + 8, acc1);
    out = 0.0F;
    for (int i = 0; i < 16; ++i) {
        out += buf[i];
    }
}

#endif  // AVX2

#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))

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
    for (int i = 0; i < 16; ++i) {
        sum += buf[i];
    }
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
    for (int i = 0; i < 16; ++i) {
        sum += buf[i];
    }
    if (b < nb) {
        sum += q41i_blk_tail_avx2(row + b * row_b, xq, b);
    }
    out = sum + msum;
}

// Q4_K x Q8: 256 weights / 144B superblock, 8 sub-blocks of 32 with
// per-sub-block 6-bit scale/min. Two sub-blocks share one 32B qs group
// (low nibbles = even sb, high = odd), so one zmm maddubs covers 64
// weights. Requires in % 256 == 0.
// term_sb = (d*d6) * (xq.d*S) - (dmin*m6) * xq.c
void q4ki_row_avx512(const uint8_t *row, size_t in, const XQuant &xq, float &out)
{
    const size_t nsb = in / 256;
    const __m512i ones = _mm512_set1_epi16(1);
    const __m256i nib_mask = _mm256_set1_epi8(0x0F);
    float acc = 0.0F;

    for (size_t s = 0; s < nsb; ++s) {
        const uint8_t *blk = row + s * 144;
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        const float dmin = fp16_to_fp32_fast(load_fp16(blk + 2));
        const uint8_t *sc = blk + 4;
        const uint8_t *qs = blk + 16;
        int d6[8], m6[8];
        for (int sb = 0; sb < 8; ++sb) {
            q4k_scale_min(sc, sb, d6[sb], m6[sb]);
        }
        for (int g = 0; g < 4; ++g) {
            // 32 packed bytes -> 64 u8 nibbles: [lo(32) | hi(32)]
            const __m256i qb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qs + g * 32));
            const __m256i lo = _mm256_and_si256(qb, nib_mask);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), nib_mask);
            __m512i w = _mm512_castsi256_si512(lo);
            w = _mm512_inserti64x4(w, hi, 1);
            const size_t xoff = (s * 8 + 2 * g) * 32;
            const __m512i xqv = _mm512_loadu_si512(reinterpret_cast<const void *>(xq.q.data() + xoff));
            const __m512i prod = _mm512_maddubs_epi16(w, xqv);
            const __m512i p32 = _mm512_madd_epi16(prod, ones);
            const float slo = static_cast<float>(hsum256_epi32(_mm512_castsi512_si256(p32)));
            const float shi = static_cast<float>(hsum256_epi32(_mm512_extracti32x8_epi32(p32, 1)));
            const int sb0 = 2 * g;
            acc += (d * static_cast<float>(d6[sb0])) * (xq.d[s * 8 + sb0] * slo) -
                   (dmin * static_cast<float>(m6[sb0])) * xq.c[s * 8 + sb0];
            acc += (d * static_cast<float>(d6[sb0 + 1])) * (xq.d[s * 8 + sb0 + 1] * shi) -
                   (dmin * static_cast<float>(m6[sb0 + 1])) * xq.c[s * 8 + sb0 + 1];
        }
    }
    out = acc;
}

// Q8_0 x Q8 (AVX-512): same madd_epi16 scheme as the AVX2 variant but one
// zmm covers a whole 32-weight block; four float accumulators break the
// fmadd dependency chain across blocks.
void q80i_row_avx512(const uint8_t *row, size_t in, const XQuant &xq, float &out)
{
    const size_t nb = in / 32;
    __m512 acc[4];
    for (int i = 0; i < 4; ++i) {
        acc[i] = _mm512_setzero_ps();
    }

    size_t b = 0;
    for (; b + 1 < nb; b += 2) {
        const uint8_t *blk0 = row + b * 34;
        const uint8_t *blk1 = row + (b + 1) * 34;
        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk0 + 2));
        const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk1 + 2));
        const __m512i wh0 = _mm512_cvtepi8_epi16(w0);
        const __m512i wh1 = _mm512_cvtepi8_epi16(w1);
        const __m512i x0 = _mm512_loadu_si512(reinterpret_cast<const void *>(xq.q16.data() + b * 32));
        const __m512i x1 = _mm512_loadu_si512(reinterpret_cast<const void *>(xq.q16.data() + (b + 1) * 32));
        const int i0 = static_cast<int>(b & 3);
        const int i1 = static_cast<int>((b + 1) & 3);
        acc[i0] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_madd_epi16(wh0, x0)),
            _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk0)) * xq.d[b]), acc[i0]);
        acc[i1] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_madd_epi16(wh1, x1)),
            _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk1)) * xq.d[b + 1]), acc[i1]);
    }
    if (b < nb) {
        const uint8_t *blk = row + b * 34;
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + 2));
        const __m512i x = _mm512_loadu_si512(reinterpret_cast<const void *>(xq.q16.data() + b * 32));
        acc[b & 3] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_madd_epi16(_mm512_cvtepi8_epi16(w), x)),
            _mm512_set1_ps(fp16_to_fp32_fast(load_fp16(blk)) * xq.d[b]), acc[b & 3]);
    }
    const __m512 s01 = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
    out = _mm512_reduce_add_ps(s01);
}

// ---------------- AVX-512 VNNI integer-dot ----------------
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))
// vpdpbusd fuses maddubs+madd: one instruction dots 64 u8 weights x 64 i8
// activations into 16 i32 lanes (4 weights per lane), vs two instructions
// for the maddubs path. Weights must be unsigned bytes: Q4_x nibbles are
// already 0..15; Q8_0 signed int8 uses a +128 offset (safe here — the
// accumulation is i32, unlike maddubs' saturating i16).

// Odd-block tail helpers (single 32-weight block -> full contribution).
inline float q40vnni_blk_tail(const uint8_t *blk, const XQuant &xq, size_t b)
{
    return q40i_blk_tail_avx2(blk, xq, b);
}

inline float q41vnni_blk_tail(const uint8_t *blk, const XQuant &xq, size_t b)
{
    return q41i_blk_tail_avx2(blk, xq, b);
}

// Q4_0 x Q8 VNNI: two 32-weight blocks per zmm.
void q40vnni_row_avx512(const uint8_t *row, size_t in, const XQuant &xq, float &out)
{
    const size_t nb = in / 32;
    const size_t row_b = 18;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const __m128i m0F = _mm_set1_epi8(0x0F);
    float corr = 0.0F;
    size_t b = 0;

    for (; b + 1 < nb; b += 2) {
        const uint8_t *r0 = row + b * row_b;
        const uint8_t *r1 = row + (b + 1) * row_b;
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
        const __m512i xqv = _mm512_loadu_si512(reinterpret_cast<const void *>(xq.q.data() + b * 64));
        const __m512i prod = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, xqv);
        const __m256 plo = _mm256_cvtepi32_ps(_mm512_castsi512_si256(prod));
        const __m256 phi = _mm256_cvtepi32_ps(_mm512_extracti32x8_epi32(prod, 1));
        const float dw0 = fp16_to_fp32_fast(load_fp16(r0));
        const float dw1 = fp16_to_fp32_fast(load_fp16(r1));
        acc0 = _mm256_fmadd_ps(plo, _mm256_set1_ps(dw0 * xq.d[b]), acc0);
        acc1 = _mm256_fmadd_ps(phi, _mm256_set1_ps(dw1 * xq.d[b + 1]), acc1);
        corr += dw0 * xq.c[b] + dw1 * xq.c[b + 1];
    }
    alignas(32) float buf[16];
    _mm256_store_ps(buf, acc0);
    _mm256_store_ps(buf + 8, acc1);
    float sum = 0.0F;
    for (int i = 0; i < 16; ++i) {
        sum += buf[i];
    }
    if (b < nb) {
        sum += q40vnni_blk_tail(row + b * row_b, xq, b);
    }
    out = sum - 8.0F * corr;
}

// Q4_1 x Q8 VNNI: same layout as Q4_0 plus the fp16 min term.
void q41vnni_row_avx512(const uint8_t *row, size_t in, const XQuant &xq, float &out)
{
    const size_t nb = in / 32;
    const size_t row_b = 20;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
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
        const __m512i xqv = _mm512_loadu_si512(reinterpret_cast<const void *>(xq.q.data() + b * 64));
        const __m512i prod = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, xqv);
        const __m256 plo = _mm256_cvtepi32_ps(_mm512_castsi512_si256(prod));
        const __m256 phi = _mm256_cvtepi32_ps(_mm512_extracti32x8_epi32(prod, 1));
        acc0 = _mm256_fmadd_ps(plo, _mm256_set1_ps(dw0 * xq.d[b]), acc0);
        acc1 = _mm256_fmadd_ps(phi, _mm256_set1_ps(dw1 * xq.d[b + 1]), acc1);
        msum += mw0 * xq.c[b] + mw1 * xq.c[b + 1];
    }
    alignas(32) float buf[16];
    _mm256_store_ps(buf, acc0);
    _mm256_store_ps(buf + 8, acc1);
    float sum = 0.0F;
    for (int i = 0; i < 16; ++i) {
        sum += buf[i];
    }
    if (b < nb) {
        sum += q41vnni_blk_tail(row + b * row_b, xq, b);
    }
    out = sum + msum;
}

// Q8_0 x Q8 VNNI: weights are packed int8 — flip the sign bit to get the
// unsigned operand dpbusd needs, and subtract 128*sum(xq) per block.
void q80vnni_row_avx512(const uint8_t *row, size_t in, const XQuant &xq, float &out)
{
    const size_t nb = in / 32;
    const size_t row_b = 34;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const __m512i sign_flip = _mm512_set1_epi8(static_cast<char>(0x80));
    float corr = 0.0F;
    size_t b = 0;

    for (; b + 1 < nb; b += 2) {
        const uint8_t *r0 = row + b * row_b;
        const uint8_t *r1 = row + (b + 1) * row_b;
        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(r0 + 2));
        const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(r1 + 2));
        __m512i w = _mm512_castsi256_si512(w0);
        w = _mm512_inserti64x4(w, w1, 1);
        w = _mm512_xor_si512(w, sign_flip);
        const __m512i xqv = _mm512_loadu_si512(reinterpret_cast<const void *>(xq.q.data() + b * 64));
        const __m512i prod = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, xqv);
        const __m256 plo = _mm256_cvtepi32_ps(_mm512_castsi512_si256(prod));
        const __m256 phi = _mm256_cvtepi32_ps(_mm512_extracti32x8_epi32(prod, 1));
        const float dw0 = fp16_to_fp32_fast(load_fp16(r0));
        const float dw1 = fp16_to_fp32_fast(load_fp16(r1));
        acc0 = _mm256_fmadd_ps(plo, _mm256_set1_ps(dw0 * xq.d[b]), acc0);
        acc1 = _mm256_fmadd_ps(phi, _mm256_set1_ps(dw1 * xq.d[b + 1]), acc1);
        corr += 128.0F * (dw0 * xq.c[b] + dw1 * xq.c[b + 1]);
    }
    alignas(32) float buf[16];
    _mm256_store_ps(buf, acc0);
    _mm256_store_ps(buf + 8, acc1);
    float sum = 0.0F;
    for (int i = 0; i < 16; ++i) {
        sum += buf[i];
    }
    if (b < nb) {
        // Odd tail block: zero-extended zmm, dpbusd low lanes only.
        const uint8_t *blk = row + b * row_b;
        const __m256i w32 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + 2));
        __m512i w = _mm512_inserti64x4(_mm512_setzero_si512(), w32, 0);
        w = _mm512_xor_si512(w, sign_flip);
        __m256i x32 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq.q.data() + b * 32));
        const __m512i xqv = _mm512_inserti64x4(_mm512_setzero_si512(), x32, 0);
        const __m512i prod = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, xqv);
        const __m256 plo = _mm256_cvtepi32_ps(_mm512_castsi512_si256(prod));
        alignas(32) float pb[8];
        _mm256_store_ps(pb, plo);
        const float dw = fp16_to_fp32_fast(load_fp16(blk));
        float lane = 0.0F;
        for (int i = 0; i < 8; ++i) {
            lane += pb[i];
        }
        sum += dw * xq.d[b] * lane - 128.0F * dw * xq.c[b];
    }
    out = sum - corr;
}

#endif  // AVX512-VNNI

#endif  // AVX-512

// ---------------- ARM NEON kernels ----------------

#if defined(_M_ARM64) || defined(__ARM_NEON) || defined(__aarch64__)

// Widen 16 u8 lanes to four float32x4_t, preserving order
// (out[0] = lanes 0..3, ..., out[3] = lanes 12..15).
inline void u8x16_to_f32x4(uint8x16_t v, float32x4_t out[4])
{
    const uint16x8_t lo16 = vmovl_u8(vget_low_u8(v));
    const uint16x8_t hi16 = vmovl_u8(vget_high_u8(v));
    out[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
    out[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
    out[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
    out[3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));
}

// Same widening for 16 signed i8 lanes.
inline void i8x16_to_f32x4(int8x16_t v, float32x4_t out[4])
{
    const int16x8_t lo16 = vmovl_s8(vget_low_s8(v));
    const int16x8_t hi16 = vmovl_s8(vget_high_s8(v));
    out[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
    out[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
    out[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
    out[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
}

// Reduce eight float32x4 accumulators (each covering a contiguous
// 4-lane slice of the row) to a scalar.
inline float neon_hsum8(const float32x4_t acc[8])
{
    alignas(16) float buf[32];
    for (int i = 0; i < 8; ++i) {
        vst1q_f32(buf + i * 4, acc[i]);
    }
    float s = 0.0f;
    for (int i = 0; i < 32; ++i) {
        s += buf[i];
    }
    return s;
}

void q40_row_neon(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t full_blocks = in / 32;
    float32x4_t acc[8];
    for (int i = 0; i < 8; ++i) {
        acc[i] = vdupq_n_f32(0.0f);
    }
    const uint8x16_t nib_lo = vdupq_n_u8(0x0F);
    const float32x4_t eight = vdupq_n_f32(8.0f);

    size_t k = 0;
    for (size_t b = 0; b < full_blocks; ++b, k += 32) {
        const uint8_t *blk = row + b * 18;
        const float32x4_t d = vdupq_n_f32(fp16_to_fp32_fast(load_fp16(blk)));
        const uint8x16_t q = vld1q_u8(blk + 2);
        float32x4_t w[8];
        u8x16_to_f32x4(vandq_u8(q, nib_lo), w);                     // weights 0..15
        u8x16_to_f32x4(vandq_u8(vshrq_n_u8(q, 4), nib_lo), w + 4);  // weights 16..31
        for (int i = 0; i < 8; ++i) {
            const float32x4_t wf = vmulq_f32(vsubq_f32(w[i], eight), d);
            acc[i] = vfmaq_f32(acc[i], wf, vld1q_f32(x + k + i * 4));
        }
    }
    out = neon_hsum8(acc);
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 18;
        out += q40_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk))) * x[e];
    }
}

void q41_row_neon(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t full_blocks = in / 32;
    float32x4_t acc[8];
    for (int i = 0; i < 8; ++i) {
        acc[i] = vdupq_n_f32(0.0f);
    }
    const uint8x16_t nib_lo = vdupq_n_u8(0x0F);

    size_t k = 0;
    for (size_t b = 0; b < full_blocks; ++b, k += 32) {
        const uint8_t *blk = row + b * 20;
        const float32x4_t d = vdupq_n_f32(fp16_to_fp32_fast(load_fp16(blk)));
        const float32x4_t m = vdupq_n_f32(fp16_to_fp32_fast(load_fp16(blk + 2)));
        const uint8x16_t q = vld1q_u8(blk + 4);
        float32x4_t w[8];
        u8x16_to_f32x4(vandq_u8(q, nib_lo), w);
        u8x16_to_f32x4(vandq_u8(vshrq_n_u8(q, 4), nib_lo), w + 4);
        for (int i = 0; i < 8; ++i) {
            const float32x4_t wf = vfmaq_f32(m, w[i], d);  // nib*d + m
            acc[i] = vfmaq_f32(acc[i], wf, vld1q_f32(x + k + i * 4));
        }
    }
    out = neon_hsum8(acc);
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 20;
        out += q41_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk)), fp16_to_fp32_fast(load_fp16(blk + 2))) * x[e];
    }
}

void q80_row_neon(const uint8_t *row, size_t in, const float *x, float &out)
{
    const size_t full_blocks = in / 32;
    float32x4_t acc[8];
    for (int i = 0; i < 8; ++i) {
        acc[i] = vdupq_n_f32(0.0f);
    }

    size_t k = 0;
    for (size_t b = 0; b < full_blocks; ++b, k += 32) {
        const uint8_t *blk = row + b * 34;
        const float32x4_t d = vdupq_n_f32(fp16_to_fp32_fast(load_fp16(blk)));
        float32x4_t w[8];
        i8x16_to_f32x4(vld1q_s8(reinterpret_cast<const int8_t *>(blk + 2)), w);       // qs[0..15]
        i8x16_to_f32x4(vld1q_s8(reinterpret_cast<const int8_t *>(blk + 18)), w + 4);  // qs[16..31]
        for (int i = 0; i < 8; ++i) {
            const float32x4_t wf = vmulq_f32(w[i], d);
            acc[i] = vfmaq_f32(acc[i], wf, vld1q_f32(x + k + i * 4));
        }
    }
    out = neon_hsum8(acc);
    for (size_t e = in - (in % 32); e < in; ++e) {
        const uint8_t *blk = row + (e / 32) * 34;
        out += q80_elem(blk, e % 32, fp16_to_fp32_fast(load_fp16(blk))) * x[e];
    }
}

#endif  // ARM NEON

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

// Scalar fallback for native Q5_K rows (raw 176B superblocks, in%256==0).
// w = d*sc*q5 - dmin*m, q5 in [0,31] = low4 + highbit*16.
void q5k_row_scalar(const uint8_t *row, size_t in, const float *x, float &out)
{
    out = 0.0f;
    const size_t nsb = in / 256;
    for (size_t s = 0; s < nsb; ++s) {
        const uint8_t *blk = row + s * 176;
        const float d = fp16_to_fp32_fast(load_fp16(blk));
        const float dmin = fp16_to_fp32_fast(load_fp16(blk + 2));
        const uint8_t *scales = blk + 4;
        const uint8_t *qh = blk + 16;
        const uint8_t *ql = blk + 48;
        const size_t base = s * 256;
        for (int sb = 0; sb < 8; ++sb) {
            const int g = sb >> 1;     // 64-weight group (0..3)
            const bool hi = (sb & 1);  // high nibble within group
            uint8_t sc, m;
            if (sb < 4) {
                sc = scales[sb] & 63;
                m = scales[sb + 4] & 63;
            } else {
                sc = (scales[sb + 4] & 0xF) | ((scales[sb - 4] >> 6) << 4);
                m = (scales[sb + 4] >> 4) | ((scales[sb] >> 6) << 4);
            }
            const float D = d * static_cast<float>(sc);
            const float M = dmin * static_cast<float>(m);
            const int bit_lo = 2 * g;
            const uint8_t *qb = ql + g * 32;
            for (int l = 0; l < 32; ++l) {
                const int q4 = hi ? (qb[l] >> 4) : (qb[l] & 0xF);
                const int hb = (qh[l] >> (hi ? bit_lo + 1 : bit_lo)) & 1;
                const float w = D * static_cast<float>(q4 + hb * 16) - M;
                out += w * x[base + sb * 32 + l];
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
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))
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
            case quant::QuantType::Q4_0:
                return q40_row_avx512;
            case quant::QuantType::Q4_1:
                return q41_row_avx512;
            case quant::QuantType::Q8_0:
                return q80_row_avx512;
            case quant::QuantType::Q4_K:
                return q4k_row_avx512;
            default:
                break;
        }
    }
#endif
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))
    if (info.has_avx2) {
        switch (type) {
            case quant::QuantType::Q4_0:
                return q40_row_avx2;
            case quant::QuantType::Q4_1:
                return q41_row_avx2;
            case quant::QuantType::Q8_0:
                return q80_row_avx2;
            case quant::QuantType::Q4_K:
                return q4k_row_avx2;
            case quant::QuantType::Q5_K:
                return q5k_row_avx2;
            default:
                break;
        }
    }
#endif
#if defined(_M_ARM64) || defined(__ARM_NEON) || defined(__aarch64__)
    switch (type) {
        case quant::QuantType::Q4_0:
            return q40_row_neon;
        case quant::QuantType::Q4_1:
            return q41_row_neon;
        case quant::QuantType::Q8_0:
            return q80_row_neon;
        default:
            break;
    }
#endif
    switch (type) {
        case quant::QuantType::Q4_0:
            return q40_row_scalar;
        case quant::QuantType::Q4_1:
            return q41_row_scalar;
        case quant::QuantType::Q8_0:
            return q80_row_scalar;
        case quant::QuantType::Q4_K:
            return q4k_row_scalar;
        case quant::QuantType::Q5_K:
            return q5k_row_scalar;
        default:
            return nullptr;
    }
}

}  // namespace

/// y[j] = dot(dequant(W_row_j), x[:]) for all N output rows.
/// W is packed ggml-style rows [N][row_bytes]; x has `in` floats; y has `out` floats.
void matvec_q(const quant::QuantWeight &w, const float *x, float *y)
{
    if (!w.valid() || x == nullptr || y == nullptr) {
        return;
    }
    const size_t K = w.in;
    const size_t N = w.out;

    // Integer-dot path (ggml-style): Q4_0/Q4_1/Q8_0/Q4_K weights x int8-
    // quantized activation. Fewer uops/byte than the fp dequant path, but
    // on CPUs without AVX512-VNNI it measured parity-or-worse (DRAM-bound).
    // Enable with SERIUS_INTDOT=1 to experiment.
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))
    static const bool yes_intdot = [] {
#if defined(_MSC_VER)
        char buf[8];
        size_t len = 0;
        return getenv_s(&len, buf, sizeof(buf), "SERIUS_INTDOT") == 0 && len > 0;
#else
        return std::getenv("SERIUS_INTDOT") != nullptr;
#endif
    }();
    const bool intdot_type = (w.type == quant::QuantType::Q4_0 || w.type == quant::QuantType::Q4_1 ||
                              w.type == quant::QuantType::Q8_0 || w.type == quant::QuantType::Q4_K);
    const bool intdot_ok = intdot_type && (K % 32) == 0 && K >= 32 &&
                           (w.type != quant::QuantType::Q4_K || (K % 256) == 0);
    if (yes_intdot && intdot_ok) {
        const auto info = get_cpu_info();
        using IRowFn = void (*)(const uint8_t *, size_t, const XQuant &, float &);
        IRowFn ifn = nullptr;
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))
        static const bool no_avx512 = [] {
#if defined(_MSC_VER)
            char buf[8];
            size_t len = 0;
            return getenv_s(&len, buf, sizeof(buf), "SERIUS_NO_AVX512") == 0 && len > 0;
#else
            return std::getenv("SERIUS_NO_AVX512") != nullptr;
#endif
        }();
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))
        // VNNI kernels measured SLOWER than the maddubs/madd_epi16 path on
        // Rocket Lake (i9-11900H, 2026-11-24 micro-bench) — keep them behind
        // an explicit opt-in for re-testing on other uarchs (e.g. SPR+).
        static const bool yes_vnni = [] {
#if defined(_MSC_VER)
            char buf[8];
            size_t len = 0;
            return getenv_s(&len, buf, sizeof(buf), "SERIUS_VNNI") == 0 && len > 0;
#else
            return std::getenv("SERIUS_VNNI") != nullptr;
#endif
        }();
#endif
        if (info.has_avx512f && !no_avx512) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))
            if (info.has_avx512_vnni && yes_vnni) {
                if (w.type == quant::QuantType::Q4_0) {
                    ifn = q40vnni_row_avx512;
                } else if (w.type == quant::QuantType::Q4_1) {
                    ifn = q41vnni_row_avx512;
                } else if (w.type == quant::QuantType::Q8_0) {
                    ifn = q80vnni_row_avx512;
                }
            }
#endif
            if (ifn == nullptr) {
                if (w.type == quant::QuantType::Q4_0) {
                    ifn = q40i_row_avx512;
                } else if (w.type == quant::QuantType::Q4_1) {
                    ifn = q41i_row_avx512;
                } else if (w.type == quant::QuantType::Q8_0) {
                    ifn = q80i_row_avx512;
                } else {
                    ifn = q4ki_row_avx512;
                }
            }
        }
#endif
        if (ifn == nullptr) {
            if (w.type == quant::QuantType::Q4_K || !info.has_avx2) {
                goto fp_path;
            }
            if (w.type == quant::QuantType::Q4_0) {
                ifn = q40i_row_avx2;
            } else if (w.type == quant::QuantType::Q4_1) {
                ifn = q41i_row_avx2;
            } else {
                ifn = q80i_row_avx2;
            }
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
    if (fn == nullptr) {
        return;
    }

    const size_t row_bytes = w.row_bytes();
    const uint8_t *base = w.bytes.data();

    // Q4_0 + AVX-512: two rows per iteration for more memory-level parallelism.
#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(SERIUS_SIMD_X86))
    if (fn == q40_row_avx512) {
        const int64_t pairs = static_cast<int64_t>(N / 2);
#pragma omp parallel for schedule(static) if (pairs >= 128)
        for (int64_t p = 0; p < pairs; ++p) {
            const size_t r0 = static_cast<size_t>(2 * p);
            const size_t r1 = r0 + 1;
            q40_row2_avx512(base + r0 * row_bytes, base + r1 * row_bytes, K, x, y[r0], y[r1]);
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
    if (!w.valid()) {
        return;
    }
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
