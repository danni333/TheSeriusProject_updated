#include "gguf_loader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "../transformer/config.h"

namespace serius {
namespace io {

namespace {

using int4_t = std::int8_t;

constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" little-endian

#pragma pack(push, 1)
struct GGUFHeader
{
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};
#pragma pack(pop)

float half_to_float(uint16_t half_val)
{
    const uint32_t sign = (static_cast<uint32_t>(half_val) >> 15) & 0x1u;
    const uint32_t exp = (static_cast<uint32_t>(half_val) >> 10) & 0x1Fu;
    const uint32_t mant = static_cast<uint32_t>(half_val) & 0x3FFu;

    if (exp == 0u && mant != 0u) {
        // fp16 subnormal: value = mant * 2^-24 (exact in fp32).
        float v = static_cast<float>(mant) * 0x1p-24f;
        return sign ? -v : v;
    }
    uint32_t bits = 0;
    if (exp == 0u) {
        bits = (sign << 31);
    } else if (exp == 0x1Fu) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// Truncating float -> IEEE 754 half (used for Q4_K->Q4_1 scale repacking).
uint16_t f32_to_f16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = static_cast<int>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

void dequantize_q4_0(const uint8_t *data, size_t n, float *out)
{
    for (size_t i = 0; i < n; i += 32) {
        uint16_t scale = data[0] | (data[1] << 8);
        float scale_f = half_to_float(scale);
        const uint8_t *q = data + 2;
        for (size_t j = 0; j < 16; ++j) {
            int4_t v0 = static_cast<int4_t>(q[j] & 0x0F) - 8;
            int4_t v1 = static_cast<int4_t>((q[j] >> 4) & 0x0F) - 8;
            if (i + j < n) {
                out[i + j] = v0 * scale_f;
            }
            if (i + j + 16 < n) {
                out[i + j + 16] = v1 * scale_f;
            }
        }
        data += 18;
    }
}

void dequantize_q4_1(const uint8_t *data, size_t n, float *out)
{
    for (size_t i = 0; i < n; i += 32) {
        uint16_t d = data[0] | (data[1] << 8);
        uint16_t m = data[2] | (data[3] << 8);
        float d_f = half_to_float(d);
        float m_f = half_to_float(m);
        const uint8_t *q = data + 4;
        for (size_t j = 0; j < 16; ++j) {
            int4_t v0 = static_cast<int4_t>(q[j] & 0x0F);
            int4_t v1 = static_cast<int4_t>((q[j] >> 4) & 0x0F);
            if (i + j < n) {
                out[i + j] = v0 * d_f + m_f;
            }
            if (i + j + 16 < n) {
                out[i + j + 16] = v1 * d_f + m_f;
            }
        }
        data += 20;
    }
}

void dequantize_q8_0(const uint8_t *data, size_t n, float *out)
{
    for (size_t i = 0; i < n; i += 32) {
        uint16_t scale = data[0] | (data[1] << 8);
        float scale_f = half_to_float(scale);
        const int8_t *q = reinterpret_cast<const int8_t *>(data + 2);
        for (size_t j = 0; j < 32 && i + j < n; ++j) {
            out[i + j] = q[j] * scale_f;
        }
        data += 34;
    }
}

// ggml block_q5_0: 32 weights per 22-byte block { fp16 d, uint32 qh, uint8 qs[16] }.
// qs[j] low nibble = weight j, high nibble = weight j+16; qh bit j / j+16 is the
// 5th bit. w = d * (q5 - 16).
void dequantize_q5_0(const uint8_t *data, size_t n, float *out)
{
    for (size_t i = 0; i < n; i += 32) {
        const float d_f = half_to_float(static_cast<uint16_t>(data[0] | (data[1] << 8)));
        uint32_t qh;
        std::memcpy(&qh, data + 2, sizeof(qh));
        const uint8_t *qs = data + 6;
        for (size_t j = 0; j < 16; ++j) {
            const int v0 = (qs[j] & 0x0F) | (static_cast<int>((qh >> j) & 1) << 4);
            const int v1 = (qs[j] >> 4) | (static_cast<int>((qh >> (j + 16)) & 1) << 4);
            if (i + j < n) {
                out[i + j] = (v0 - 16) * d_f;
            }
            if (i + j + 16 < n) {
                out[i + j + 16] = (v1 - 16) * d_f;
            }
        }
        data += 22;
    }
}

// ggml block_q4_K: 256 weights per 144-byte superblock.
// { fp16 d, fp16 dmin, uint8 scales[12], uint8 qs[128] }
// 8 sub-blocks of 32: w = d*sc[j]*nibble - dmin*m[j].
inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

void dequantize_q4_k(const uint8_t *data, size_t n, float *out)
{
    const size_t nb = n / 256;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = data + i * 144;
        const float d = half_to_float(static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
        const float dmin = half_to_float(static_cast<uint16_t>(blk[2] | (blk[3] << 8)));
        const uint8_t *scales = blk + 4;
        const uint8_t *q = blk + 16;
        float *y = out + i * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * (q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

// ggml block_q6_K: 256 weights per 210-byte superblock.
// { uint8 ql[128], uint8 qh[64], int8 scales[16], fp16 d } — d is LAST.
// Q5_K: 256 weights per 176-byte superblock.
// Layout: d(f16) dmin(f16) scales[12] qh[32] qs[128].
// x = d*sc*q - min*m  with q in [0,31] (4 low bits + 1 high bit).
void dequantize_q5_k(const uint8_t *data, size_t n, float *out)
{
    const size_t nb = n / 256;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = data + i * 176;
        const float d = half_to_float(static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
        const float dmin = half_to_float(static_cast<uint16_t>(blk[2] | (blk[3] << 8)));
        const uint8_t *scales = blk + 4;
        const uint8_t *qh = blk + 16;
        const uint8_t *ql = blk + 48;
        float *y = out + i * 256;

        auto scale_min = [&](int j, uint8_t &sc, uint8_t &m) {
            if (j < 4) {
                sc = scales[j] & 63;
                m = scales[j + 4] & 63;
            } else {
                sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
                m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            }
        };

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            scale_min(is + 0, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            scale_min(is + 1, sc, m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void dequantize_q6_k(const uint8_t *data, size_t n, float *out)
{
    const size_t nb = n / 256;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = data + i * 210;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t *sc = reinterpret_cast<const int8_t *>(blk + 192);
        const float d = half_to_float(static_cast<uint16_t>(blk[208] | (blk[209] << 8)));
        float *y = out + i * 256;
        for (int half = 0; half < 2; ++half) {
            const uint8_t *ql_h = ql + half * 64;
            const uint8_t *qh_h = qh + half * 32;
            const int8_t *sc_h = sc + half * 8;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = static_cast<int8_t>((ql_h[l] & 0xF) | (((qh_h[l] >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int8_t>((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int8_t>((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int8_t>((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc_h[is + 0] * q1;
                y[l + 32] = d * sc_h[is + 2] * q2;
                y[l + 64] = d * sc_h[is + 4] * q3;
                y[l + 96] = d * sc_h[is + 6] * q4;
            }
            y += 128;
        }
    }
}

// ggml block_q5_1: 32 weights per 24-byte block { fp16 d, fp16 m, uint32 qh,
// uint8 qs[16] }. Same layout as Q5_0 but with a min offset instead of the
// -16 bias: w = d*q5 + m, q5 in [0,31].
void dequantize_q5_1(const uint8_t *data, size_t n, float *out)
{
    for (size_t i = 0; i < n; i += 32) {
        const float d_f = half_to_float(static_cast<uint16_t>(data[0] | (data[1] << 8)));
        const float m_f = half_to_float(static_cast<uint16_t>(data[2] | (data[3] << 8)));
        uint32_t qh;
        std::memcpy(&qh, data + 4, sizeof(qh));
        const uint8_t *qs = data + 8;
        for (size_t j = 0; j < 16; ++j) {
            const int v0 = (qs[j] & 0x0F) | (static_cast<int>((qh >> j) & 1) << 4);
            const int v1 = (qs[j] >> 4) | (static_cast<int>((qh >> (j + 16)) & 1) << 4);
            if (i + j < n) {
                out[i + j] = v0 * d_f + m_f;
            }
            if (i + j + 16 < n) {
                out[i + j + 16] = v1 * d_f + m_f;
            }
        }
        data += 24;
    }
}

// ggml block_q2_K: 256 weights per 84-byte superblock.
// { uint8 scales[16], uint8 qs[64], fp16 d, fp16 dmin }
// 16 sub-blocks of 16: w = d*sc*q - dmin*m, q in {0..3}.
void dequantize_q2_k(const uint8_t *data, size_t n, float *out)
{
    const size_t nb = n / 256;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = data + i * 84;
        const uint8_t *scales = blk;
        const uint8_t *q = blk + 16;
        const float d = half_to_float(static_cast<uint16_t>(blk[80] | (blk[81] << 8)));
        const float dmin = half_to_float(static_cast<uint16_t>(blk[82] | (blk[83] << 8)));
        float *y = out + i * 256;
        int is = 0;
        for (int nn = 0; nn < 256; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                const float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<int8_t>((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                const float dl2 = d * (sc & 0xF), ml2 = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl2 * static_cast<int8_t>((q[l + 16] >> shift) & 3) - ml2;
                shift += 2;
            }
            q += 32;
        }
    }
}

// ggml block_q8_K (GGUF type 15): 256 weights per 292-byte superblock
// { float d, int8 qs[256], int16 bsums[16] }. bsums are only used by the
// quantized dot-product kernels; scalar dequant ignores them.
void dequantize_q8_k(const uint8_t *data, size_t n, float *out)
{
    const size_t nb = n / 256;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = data + i * 292;
        float d;
        std::memcpy(&d, blk, sizeof(d));
        const int8_t *qs = reinterpret_cast<const int8_t *>(blk + 4);
        float *y = out + i * 256;
        for (int j = 0; j < 256; ++j) {
            *y++ = d * static_cast<float>(qs[j]);
        }
    }
}

// ggml block_q3_K: 256 weights per 110-byte superblock.
// { uint8 hmask[32], uint8 qs[64], uint8 scales[12], fp16 d }
// w = d*(sc-32)*((q2 | hbit<<2) - 4), q2 in {0..3}.
void dequantize_q3_k(const uint8_t *data, size_t n, float *out)
{
    const size_t nb = n / 256;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = data + i * 110;
        const float d_all = half_to_float(static_cast<uint16_t>(blk[108] | (blk[109] << 8)));
        const uint8_t *q = blk;   // qs at offset 32
        const uint8_t *hm = blk;  // hmask at offset 0
        q += 32;
        const uint8_t *scales12 = blk + 96;

        uint32_t aux[4];
        std::memcpy(aux, scales12, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = reinterpret_cast<const int8_t *>(aux);

        float *y = out + i * 256;
        uint8_t m = 1;
        int is = 0;
        for (int nn = 0; nn < 256; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                const float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    const int qv = static_cast<int8_t>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                    *y++ = dl * qv;
                }
                const float dl2 = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    const int qv = static_cast<int8_t>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                    *y++ = dl2 * qv;
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// ggml block_tq1_0: 256 weights per 54-byte superblock { uint8 qs[48],
// uint8 qh[4], fp16 d }. 5 ternary digits per byte via powers of 3.
void dequantize_tq1_0(const uint8_t *data, size_t n, float *out)
{
    const size_t nb = n / 256;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = data + i * 54;
        const uint8_t *qs = blk;
        const uint8_t *qh = blk + 48;
        const float d = half_to_float(static_cast<uint16_t>(blk[52] | (blk[53] << 8)));
        float *y = out + i * 256;
        for (size_t j = 0; j < 32; j += 32) {
            for (int nn = 0; nn < 5; ++nn) {
                for (size_t mm = 0; mm < 32; ++mm) {
                    const uint8_t q = qs[j + mm] * pow3[nn];
                    const int16_t xi = static_cast<int16_t>((static_cast<uint16_t>(q) * 3) >> 8);
                    *y++ = static_cast<float>(xi - 1) * d;
                }
            }
        }
        for (size_t j = 32; j < 48; j += 16) {
            for (int nn = 0; nn < 5; ++nn) {
                for (size_t mm = 0; mm < 16; ++mm) {
                    const uint8_t q = qs[j + mm] * pow3[nn];
                    const int16_t xi = static_cast<int16_t>((static_cast<uint16_t>(q) * 3) >> 8);
                    *y++ = static_cast<float>(xi - 1) * d;
                }
            }
        }
        for (int nn = 0; nn < 4; ++nn) {
            for (size_t j = 0; j < 4; ++j) {
                const uint8_t q = qh[j] * pow3[nn];
                const int16_t xi = static_cast<int16_t>((static_cast<uint16_t>(q) * 3) >> 8);
                *y++ = static_cast<float>(xi - 1) * d;
            }
        }
    }
}

// ggml block_tq2_0: 256 weights per 66-byte superblock { uint8 qs[64],
// fp16 d }. 2-bit quants offset by -1.
void dequantize_tq2_0(const uint8_t *data, size_t n, float *out)
{
    const size_t nb = n / 256;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = data + i * 66;
        const uint8_t *qs = blk;
        const float d = half_to_float(static_cast<uint16_t>(blk[64] | (blk[65] << 8)));
        float *y = out + i * 256;
        for (size_t j = 0; j < 64; j += 32) {
            for (int l = 0; l < 4; ++l) {
                for (size_t mm = 0; mm < 32; ++mm) {
                    const int8_t q = static_cast<int8_t>((qs[j + mm] >> (l * 2)) & 3);
                    *y++ = static_cast<float>(q - 1) * d;
                }
            }
        }
    }
}

// ggml kvalues_iq4nl: non-linear 4-bit codebook.
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// ggml block_iq4_nl: 32 weights per 18-byte block { fp16 d, uint8 qs[16] }.
void dequantize_iq4_nl(const uint8_t *data, size_t n, float *out)
{
    const size_t nb = n / 32;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = data + i * 18;
        const float d = half_to_float(static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        float *y = out + i * 32;
        for (int j = 0; j < 16; ++j) {
            y[j] = d * kvalues_iq4nl[qs[j] & 0xF];
            y[j + 16] = d * kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

// ggml block_iq4_xs: 256 weights per 136-byte superblock
// { fp16 d, uint16 scales_h, uint8 scales_l[4], uint8 qs[128] }.
// 8 sub-blocks of 32: dl = d*(ls-32), ls 6-bit packed across scales_h/l.
void dequantize_iq4_xs(const uint8_t *data, size_t n, float *out)
{
    const size_t nb = n / 256;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = data + i * 136;
        const float d = half_to_float(static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
        const uint16_t scales_h = static_cast<uint16_t>(blk[2] | (blk[3] << 8));
        const uint8_t *scales_l = blk + 4;
        const uint8_t *qs = blk + 8;
        float *y = out + i * 256;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j] = dl * kvalues_iq4nl[qs[j] & 0xF];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

// Convert packed Q4_K superblocks into QuantWeight rows of Q4_1 blocks
// (w = nibble*D + M per 32 weights). Q4_K sub-block sb uses the low nibble of
// qs[(sb/2)*32 + l] when sb is even, the high nibble when odd.
void q4_k_to_q41(const uint8_t *packed, size_t numel, size_t in, size_t out, quant::QuantWeight &qw)
{
    if (numel % 256 != 0 || in % 256 != 0 || in == 0) {
        return;
    }
    const size_t sbs_per_row = in / 256;
    const size_t bpr = in / 32;  // Q4_1 blocks per row
    qw.type = quant::QuantType::Q4_1;
    qw.in = in;
    qw.out = out;
    qw.bytes.assign(out * bpr * 20, 0);
    for (size_t r = 0; r < out; ++r) {
        const uint8_t *src_row = packed + r * sbs_per_row * 144;
        uint8_t *dst_row = qw.bytes.data() + r * bpr * 20;
        for (size_t s = 0; s < sbs_per_row; ++s) {
            const uint8_t *blk = src_row + s * 144;
            const float d = half_to_float(static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
            const float dmin = half_to_float(static_cast<uint16_t>(blk[2] | (blk[3] << 8)));
            const uint8_t *scales = blk + 4;
            const uint8_t *qs = blk + 16;
            for (int sb = 0; sb < 8; ++sb) {
                uint8_t sc, m;
                get_scale_min_k4(sb, scales, &sc, &m);
                const float D = d * sc;
                const float M = -(dmin * m);
                const uint8_t *qb = qs + (sb / 2) * 32;
                const bool hi = (sb & 1) != 0;
                uint8_t *ob = dst_row + (s * 8 + sb) * 20;
                // fp16 D and M
                const uint16_t d16 = f32_to_f16(D);
                const uint16_t m16 = f32_to_f16(M);
                ob[0] = static_cast<uint8_t>(d16 & 0xFF);
                ob[1] = static_cast<uint8_t>(d16 >> 8);
                ob[2] = static_cast<uint8_t>(m16 & 0xFF);
                ob[3] = static_cast<uint8_t>(m16 >> 8);
                // repack nibbles: byte j = w[j] | (w[j+16]<<4)
                for (int j = 0; j < 16; ++j) {
                    const uint8_t lo = hi ? (qb[j] >> 4) : (qb[j] & 0xF);
                    const uint8_t hi_n = hi ? (qb[j + 16] >> 4) : (qb[j + 16] & 0xF);
                    ob[4 + j] = static_cast<uint8_t>(lo | (hi_n << 4));
                }
            }
        }
    }
}

// Requantize a row-major FP32 [out][in] matrix into Q4_1 QuantWeight rows
// (w = nibble*D + M per 32 weights). Used for K-quant tensors (Q6_K) that have
// no native packed representation in our kernels.
void quantize_f32_to_q41(const float *w, size_t in, size_t out, quant::QuantWeight &qw)
{
    if (in == 0 || out == 0 || (in % 32) != 0) {
        return;
    }
    const size_t bpr = in / 32;
    qw.type = quant::QuantType::Q4_1;
    qw.in = in;
    qw.out = out;
    qw.bytes.assign(out * bpr * 20, 0);
    for (size_t r = 0; r < out; ++r) {
        const float *wr = w + r * in;
        uint8_t *dr = qw.bytes.data() + r * bpr * 20;
        for (size_t b = 0; b < bpr; ++b) {
            const float *wb = wr + b * 32;
            float mn = wb[0], mx = wb[0];
            for (int j = 1; j < 32; ++j) {
                mn = mn < wb[j] ? mn : wb[j];
                mx = mx > wb[j] ? mx : wb[j];
            }
            const float d = (mx > mn) ? (mx - mn) / 15.0f : 0.0f;
            const float m = mn;
            uint8_t *ob = dr + b * 20;
            const uint16_t d16 = f32_to_f16(d);
            const uint16_t m16 = f32_to_f16(m);
            ob[0] = static_cast<uint8_t>(d16 & 0xFF);
            ob[1] = static_cast<uint8_t>(d16 >> 8);
            ob[2] = static_cast<uint8_t>(m16 & 0xFF);
            ob[3] = static_cast<uint8_t>(m16 >> 8);
            for (int j = 0; j < 16; ++j) {
                int n0 = 0, n1 = 0;
                if (d > 0.0f) {
                    n0 = static_cast<int>(std::lrintf((wb[j] - m) / d));
                    n1 = static_cast<int>(std::lrintf((wb[j + 16] - m) / d));
                    n0 = n0 < 0 ? 0 : (n0 > 15 ? 15 : n0);
                    n1 = n1 < 0 ? 0 : (n1 > 15 ? 15 : n1);
                }
                ob[4 + j] = static_cast<uint8_t>(n0 | (n1 << 4));
            }
        }
    }
}

}  // anonymous namespace

quant::QuantWeight quant_weight_q4k_to_q41(const quant::QuantWeight &w)
{
    quant::QuantWeight out;
    if (w.type != quant::QuantType::Q4_K || w.in == 0 || w.out == 0 || (w.in % 256) != 0) {
        return out;
    }
    q4_k_to_q41(w.bytes.data(), w.in * w.out, w.in, w.out, out);
    return out;
}

quant::QuantWeight quant_weight_q80_to_q41(const quant::QuantWeight &w)
{
    quant::QuantWeight out;
    if (w.type != quant::QuantType::Q8_0 || w.in == 0 || w.out == 0 || (w.in % 32) != 0) {
        return out;
    }
    const size_t bpr = w.in / 32;
    std::vector<float> f32(static_cast<size_t>(w.out) * w.in);
    for (size_t r = 0; r < w.out; ++r) {
        const uint8_t *row = w.bytes.data() + r * bpr * 34;
        float *dst = f32.data() + r * w.in;
        for (size_t b = 0; b < bpr; ++b) {
            const uint8_t *blk = row + b * 34;
            const float d = half_to_float(static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
            const int8_t *q = reinterpret_cast<const int8_t *>(blk + 2);
            for (size_t j = 0; j < 32; ++j) {
                dst[b * 32 + j] = static_cast<float>(q[j]) * d;
            }
        }
    }
    quantize_f32_to_q41(f32.data(), w.in, w.out, out);
    return out;
}

namespace {

uint64_t read_u64_le(std::ifstream &f)
{
    uint64_t value = 0;
    std::array<uint8_t, sizeof(value)> bytes{};
    f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    for (size_t i = 0; i < bytes.size(); ++i) {
        value |= (static_cast<uint64_t>(bytes[i]) << (8 * i));
    }
    return value;
}

uint32_t read_u32_le(std::ifstream &f)
{
    uint32_t value = 0;
    std::array<uint8_t, sizeof(value)> bytes{};
    f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    for (size_t i = 0; i < bytes.size(); ++i) {
        value |= (static_cast<uint32_t>(bytes[i]) << (8 * i));
    }
    return value;
}

uint16_t read_u16_le(std::ifstream &f)
{
    uint16_t value = 0;
    std::array<uint8_t, sizeof(value)> bytes{};
    f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    for (size_t i = 0; i < bytes.size(); ++i) {
        value |= (static_cast<uint16_t>(bytes[i]) << (8 * i));
    }
    return value;
}

int32_t read_i32_le(std::ifstream &f)
{
    uint32_t raw = read_u32_le(f);
    return static_cast<int32_t>(raw);
}

int64_t read_i64_le(std::ifstream &f)
{
    uint64_t raw = read_u64_le(f);
    return static_cast<int64_t>(raw);
}

// Bytes remaining in the stream from the current position. Every length read
// from the file is checked against this before it is used for an allocation,
// so a crafted GGUF cannot turn a header field into a multi-GB allocation.
uint64_t stream_remaining(std::ifstream &f)
{
    f.clear();
    const std::streampos cur = f.tellg();
    if (cur == std::streampos(-1)) {
        return 0;
    }
    f.seekg(0, std::ios::end);
    const std::streampos end = f.tellg();
    f.clear();
    f.seekg(cur);
    if (end == std::streampos(-1)) {
        return 0;
    }
    const auto span = static_cast<std::streamoff>(end) - static_cast<std::streamoff>(cur);
    return span > 0 ? static_cast<uint64_t>(span) : 0;
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

// Upper bound for a single string value in the KV section. Legit GGUF
// strings (arch names, chat templates, individual tokens) are far smaller.
constexpr uint64_t kMaxStringBytes = 256ull * 1024 * 1024;

bool read_string_kv_value(std::ifstream &f, std::string *out)
{
    uint64_t len = read_u64_le(f);
    if (!f || len > kMaxStringBytes || len > stream_remaining(f)) return false;
    std::string value(static_cast<size_t>(len), '\0');
    f.read(&value[0], static_cast<std::streamsize>(len));
    if (!f) return false;
    if (out) *out = std::move(value);
    return true;
}

bool skip_gguf_value(std::ifstream &f, uint32_t type)
{
    switch (type) {
        case 0:
            return true;
        case 1: /* uint8 */ {
            uint8_t v = 0;
            f.read(reinterpret_cast<char *>(&v), 1);
            return f.good();
        }
        case 2: /* int8 */ {
            int8_t v = 0;
            f.read(reinterpret_cast<char *>(&v), 1);
            return f.good();
        }
        case 3: /* uint16 */ {
            uint16_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            return f.good();
        }
        case 4: /* uint32 */ {
            uint32_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            return f.good();
        }
        case 5: /* int32 */ {
            int32_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            return f.good();
        }
        case 6: /* float32 */ {
            float v = 0.0f;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            return f.good();
        }
        case 7: /* bool */ {
            uint8_t v = 0;
            f.read(reinterpret_cast<char *>(&v), 1);
            return f.good();
        }
        case 8:
            return read_string_kv_value(f, nullptr);
        case 9: {
            uint32_t elem_type = read_u32_le(f);
            uint64_t n = read_u64_le(f);
            if (!f) return false;
            // Every element consumes at least 1 byte on disk (elem_type 0
            // consumes none and would spin forever), so n can never exceed
            // the remaining file size in a valid file.
            if (elem_type == 0 || n > stream_remaining(f)) return false;
            for (uint64_t i = 0; i < n && f.good(); ++i) {
                if (!skip_gguf_value(f, elem_type)) return false;
            }
            return f.good();
        }
        default: {
            static const size_t type_sizes[] = {0, 1, 1, 2, 4, 4, 4, 1, 0, 0, 4, 8};
            if (type < sizeof(type_sizes) / sizeof(type_sizes[0]) && type_sizes[type] > 0) {
                f.seekg(static_cast<std::streamoff>(type_sizes[type]), std::ios::cur);
                return f.good();
            }
            return true;
        }
    }
}

bool read_gguf_kv_entry(std::ifstream &f, std::string *key, uint32_t *type)
{
    uint64_t key_len = read_u64_le(f);
    if (!f || key_len > kMaxStringBytes || key_len > stream_remaining(f)) return false;
    std::string key_value(static_cast<size_t>(key_len), '\0');
    f.read(&key_value[0], static_cast<std::streamsize>(key_len));
    if (!f) return false;

    *type = read_u32_le(f);
    if (!f) return false;

    if (key) *key = std::move(key_value);
    return true;
}

bool read_float_value(std::ifstream &f, uint32_t type, float *out)
{
    if (type == 6) {
        float v = 0.0f;
        f.read(reinterpret_cast<char *>(&v), sizeof(v));
        if (out) *out = v;
        return f.good();
    }
    if (type == 12) {
        double v = 0.0;
        f.read(reinterpret_cast<char *>(&v), sizeof(v));
        if (out) *out = static_cast<float>(v);
        return f.good();
    }
    return false;
}

bool read_scalar_value(std::ifstream &f, uint32_t type, uint64_t *out)
{
    switch (type) {
        case 1: {
            uint8_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = v;
            return f.good();
        }
        case 2: {
            int8_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = static_cast<uint64_t>(v);
            return f.good();
        }
        case 3: {
            uint16_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = v;
            return f.good();
        }
        case 4: {
            uint32_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = v;
            return f.good();
        }
        case 5: {
            int32_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = static_cast<uint64_t>(v);
            return f.good();
        }
        case 7: {
            uint8_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = v;
            return f.good();
        }
        case 6:
        case 8:
        case 9:
        default:
            return false;
    }
}

GGUFTensorInfo read_tensor_info(std::ifstream &f)
{
    GGUFTensorInfo info;
    uint64_t name_len = read_u64_le(f);
    if (!f || name_len > kMaxStringBytes || name_len > stream_remaining(f)) {
        f.setstate(std::ios::failbit);
        return info;
    }
    std::string name(static_cast<size_t>(name_len), '\0');
    f.read(&name[0], static_cast<std::streamsize>(name_len));
    if (!f) return info;
    info.name = std::move(name);

    uint32_t n_dims = read_u32_le(f);
    // GGUF tensors have at most a handful of dims; also each dim costs 8
    // bytes on disk, so the remaining-size check bounds the allocation.
    if (!f || n_dims > 16 || static_cast<uint64_t>(n_dims) * 8 > stream_remaining(f)) {
        f.setstate(std::ios::failbit);
        return info;
    }
    info.shape.resize(n_dims);
    for (uint32_t i = 0; i < n_dims; ++i) {
        info.shape[i] = read_u64_le(f);
        if (!f) return info;
    }

    uint32_t dtype = read_u32_le(f);
    if (!f) return info;
    info.dtype = static_cast<GGMLType>(dtype);

    info.offset = read_u64_le(f);
    return info;
}

// True for every dtype the loader's dispatch below knows how to read.
bool gguf_dtype_supported(GGMLType dtype)
{
    switch (dtype) {
        case GGMLType::F32:
        case GGMLType::F16:
        case GGMLType::BF16:
        case GGMLType::Q4_0:
        case GGMLType::Q4_1:
        case GGMLType::Q8_0:
        case GGMLType::Q5_0:
        case GGMLType::Q4_K:
        case GGMLType::Q5_K:
        case GGMLType::Q6_K:
        case GGMLType::Q5_1:
        case GGMLType::Q2_K:
        case GGMLType::Q3_K:
        case GGMLType::IQ4_NL:
        case GGMLType::IQ4_XS:
        case GGMLType::TQ1_0:
        case GGMLType::TQ2_0:
        case GGMLType::Q8_K:
            return true;
        default:
            return false;
    }
}

// Bytes a tensor of `numel` elements occupies in the GGUF data section for
// the given dtype. Returns std::nullopt for unsupported dtypes, for shapes
// that violate the dtype's block divisibility, or on arithmetic overflow —
// callers must check the result before allocating anything.
std::optional<uint64_t> gguf_payload_bytes(GGMLType dtype, uint64_t numel)
{
    uint64_t block_elems = 0;
    uint64_t block_bytes = 0;
    bool exact = false;  // require numel % block_elems == 0 (matches dispatch)
    switch (dtype) {
        case GGMLType::F32:
            block_elems = 1;
            block_bytes = 4;
            break;
        case GGMLType::F16:
            block_elems = 1;
            block_bytes = 2;
            break;
        case GGMLType::BF16:
            block_elems = 1;
            block_bytes = 2;
            break;
        case GGMLType::Q4_0:
            block_elems = 32;
            block_bytes = 18;
            break;
        case GGMLType::Q4_1:
            block_elems = 32;
            block_bytes = 20;
            break;
        case GGMLType::Q8_0:
            block_elems = 32;
            block_bytes = 34;
            break;
        case GGMLType::Q5_0:
            block_elems = 32;
            block_bytes = 22;
            exact = true;
            break;
        case GGMLType::Q4_K:
            block_elems = 256;
            block_bytes = 144;
            exact = true;
            break;
        case GGMLType::Q5_K:
            block_elems = 256;
            block_bytes = 176;
            exact = true;
            break;
        case GGMLType::Q6_K:
            block_elems = 256;
            block_bytes = 210;
            exact = true;
            break;
        case GGMLType::Q5_1:
            block_elems = 32;
            block_bytes = 24;
            exact = true;
            break;
        case GGMLType::Q2_K:
            block_elems = 256;
            block_bytes = 84;
            exact = true;
            break;
        case GGMLType::Q3_K:
            block_elems = 256;
            block_bytes = 110;
            exact = true;
            break;
        case GGMLType::IQ4_NL:
            block_elems = 32;
            block_bytes = 18;
            exact = true;
            break;
        case GGMLType::IQ4_XS:
            block_elems = 256;
            block_bytes = 136;
            exact = true;
            break;
        case GGMLType::TQ1_0:
            block_elems = 256;
            block_bytes = 54;
            exact = true;
            break;
        case GGMLType::TQ2_0:
            block_elems = 256;
            block_bytes = 66;
            exact = true;
            break;
        case GGMLType::Q8_K:
            block_elems = 256;
            block_bytes = 292;
            exact = true;
            break;
        default:
            return std::nullopt;
    }
    uint64_t blocks;
    if (exact) {
        if (numel % block_elems != 0) {
            return std::nullopt;
        }
        blocks = numel / block_elems;
    } else {
        blocks = (numel + block_elems - 1) / block_elems;
    }
    uint64_t bytes = 0;
    if (!checked_mul_u64(blocks, block_bytes, &bytes)) {
        return std::nullopt;
    }
    return bytes;
}

}  // namespace

// Tensors that are critical for model functionality — loading must fail if
// their quantization is unsupported. Embedding/lm_head can sometimes fall back
// (tied weights), but attention/FFN weights are essential.
static bool is_critical_tensor(const std::string &name)
{
    // Critical: attention projections, FFN weights, output norm
    return name.find("attn_q.weight") != std::string::npos || name.find("attn_k.weight") != std::string::npos ||
           name.find("attn_v.weight") != std::string::npos || name.find("attn_output.weight") != std::string::npos ||
           name.find("ffn_gate.weight") != std::string::npos || name.find("ffn_up.weight") != std::string::npos ||
           name.find("ffn_down.weight") != std::string::npos || name.find("ffn_norm.weight") != std::string::npos ||
           name.find("attn_norm.weight") != std::string::npos || name.find("output_norm.weight") != std::string::npos ||
           name.find("norm.weight") != std::string::npos;
}

GGUFMeta load_gguf_metadata(const std::string &path)
{
    GGUFMeta out;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return out;
    }

    GGUFHeader h{};
    f.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!f || h.magic != GGUF_MAGIC) {
        return out;
    }

    if (h.version < 1 || h.version > 10) {
        return out;
    }

    out.version = h.version;
    out.n_tensors = h.n_tensors;
    out.n_kv = h.n_kv;
    out.valid = true;
    return out;
}

std::vector<std::string> gguf_tensor_names(const std::string &path)
{
    std::vector<std::string> names;
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return names;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return names;
    }

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type)) {
            break;
        }
        (void)key;
        if (!skip_gguf_value(f, type)) {
            break;
        }
    }

    for (uint64_t i = 0; i < meta.n_tensors && f; ++i) {
        GGUFTensorInfo info = read_tensor_info(f);
        if (!f || info.name.empty()) {
            break;
        }
        names.push_back(info.name);
    }
    return names;
}

GGUFWeightStats gguf_weight_stats(const std::string &path)
{
    GGUFWeightStats out;
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return out;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return out;
    }

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type)) {
            return out;
        }
        (void)key;
        if (!skip_gguf_value(f, type)) {
            return out;
        }
    }

    for (uint64_t i = 0; i < meta.n_tensors && f; ++i) {
        GGUFTensorInfo info = read_tensor_info(f);
        if (!f || info.name.empty()) {
            break;
        }
        uint64_t numel = 1;
        for (uint64_t dim : info.shape) {
            numel *= dim;
        }
        out.total_params += numel;
        if (const auto bytes = gguf_payload_bytes(info.dtype, numel)) {
            out.total_bytes += *bytes;
        }
        ++out.tensors_counted;
    }
    out.valid = f.good() || f.eof();
    return out;
}

std::optional<GGUFTensorInfo> get_tensor_info(const std::string &path, const std::string &tensor_name)
{
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return std::nullopt;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::nullopt;
    }

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type)) {
            break;
        }
        (void)key;
        if (!skip_gguf_value(f, type)) {
            break;
        }
    }

    for (uint64_t i = 0; i < meta.n_tensors; ++i) {
        GGUFTensorInfo info = read_tensor_info(f);
        if (!f) {
            break;
        }
        if (info.name == tensor_name) {
            return info;
        }
    }
    return std::nullopt;
}

std::unordered_map<std::string, core::Tensor> load_all_tensors(
    const std::string &path, std::unordered_map<std::string, quant::QuantWeight> *quant_out)
{
    std::unordered_map<std::string, core::Tensor> result;
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return result;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return result;
    }

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type)) {
            break;
        }
        (void)key;
        if (!skip_gguf_value(f, type)) {
            break;
        }
    }

    std::vector<GGUFTensorInfo> tensor_infos;
    for (uint64_t i = 0; i < meta.n_tensors; ++i) {
        GGUFTensorInfo info = read_tensor_info(f);
        if (!f || info.name.empty()) {
            break;
        }
        tensor_infos.push_back(info);
    }
    if (tensor_infos.empty()) {
        return result;
    }

    const std::streampos cur_pos = f.tellg();
    constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;
    const uint64_t cur_u64 = static_cast<uint64_t>(cur_pos);
    const uint64_t rem = cur_u64 % GGUF_DEFAULT_ALIGNMENT;
    const uint64_t data_base_u64 = (rem == 0) ? cur_u64 : (cur_u64 + (GGUF_DEFAULT_ALIGNMENT - rem));
    const std::streampos data_base = static_cast<std::streampos>(data_base_u64);

    // Total file size and bytes available for tensor payloads. Every tensor's
    // declared shape/dtype/offset is validated against this BEFORE any
    // allocation, so a crafted GGUF cannot force huge allocations or reads
    // past end-of-file.
    f.clear();
    f.seekg(0, std::ios::end);
    const auto file_end = f.tellg();
    f.clear();
    f.seekg(cur_pos);
    if (file_end == std::streampos(-1)) {
        return result;
    }
    const uint64_t file_size = static_cast<uint64_t>(static_cast<std::streamoff>(file_end));
    const uint64_t data_avail = file_size > data_base_u64 ? file_size - data_base_u64 : 0;

    // Critical tensors (attention/FFN/norm weights) must never be skipped
    // silently: a zero-weight model that "loads" is worse than a hard error.
    // Throws on critical tensors; warns and returns otherwise.
    auto skip_or_fail = [](const GGUFTensorInfo &info, const std::string &reason) {
        if (is_critical_tensor(info.name)) {
            throw std::runtime_error("[GGUF] critical tensor '" + info.name + "' " + reason);
        }
        std::fprintf(stderr, "[GGUF] tensor %s skipped: %s\n", info.name.c_str(), reason.c_str());
    };

    for (const auto &info : tensor_infos) {
        // Checked product of dims — crafted shapes must not overflow.
        uint64_t numel64 = 1;
        bool dims_bad = false;
        for (uint64_t dim : info.shape) {
            if (dim != 0 && numel64 > std::numeric_limits<uint64_t>::max() / dim) {
                dims_bad = true;
                break;
            }
            numel64 *= dim;
        }
        if (dims_bad || numel64 == 0 || numel64 > std::numeric_limits<size_t>::max()) {
            skip_or_fail(info, "has invalid shape/dimensions");
            continue;
        }

        const auto payload = gguf_payload_bytes(info.dtype, numel64);
        if (payload.has_value()) {
            if (info.offset > data_avail || *payload > data_avail - info.offset) {
                skip_or_fail(info, "payload exceeds file bounds");
                continue;
            }
        } else if (gguf_dtype_supported(info.dtype)) {
            skip_or_fail(info, "has unsupported shape for its dtype");
            continue;
        } else {
            skip_or_fail(info, "uses unsupported dtype " + std::to_string(static_cast<unsigned>(info.dtype)) +
                                   " (supported: F32, F16, BF16, Q4_0, Q4_1, Q8_0, Q5_0, Q5_1, "
                                   "Q4_K, Q5_K, Q6_K, Q8_K, Q2_K, Q3_K, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0)");
            continue;
        }

        std::vector<size_t> shape;
        shape.reserve(info.shape.size());
        for (uint64_t dim : info.shape) {
            shape.push_back(static_cast<size_t>(dim));
        }

        const size_t numel = static_cast<size_t>(numel64);
        std::vector<float> values(numel, 0.0f);
        f.seekg(data_base + static_cast<std::streamoff>(info.offset));

        if (info.dtype == GGMLType::F32) {
            f.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(numel * sizeof(float)));
        } else if (info.dtype == GGMLType::F16) {
            std::vector<uint16_t> half_values(numel);
            f.read(reinterpret_cast<char *>(half_values.data()),
                   static_cast<std::streamsize>(numel * sizeof(uint16_t)));
            for (size_t i = 0; i < numel; ++i) {
                values[i] = half_to_float(half_values[i]);
            }
        } else if (info.dtype == GGMLType::Q4_0) {
            const size_t block_count = (numel + 31) / 32;
            std::vector<uint8_t> packed(block_count * 18);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q4_0(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2) {
                quant::QuantWeight qw;
                qw.type = quant::QuantType::Q4_0;
                qw.in = shape[0];
                qw.out = shape[1];
                qw.bytes = std::move(packed);
                quant_out->emplace(info.name, std::move(qw));
            }
        } else if (info.dtype == GGMLType::Q4_1) {
            const size_t block_count = (numel + 31) / 32;
            std::vector<uint8_t> packed(block_count * 20);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q4_1(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2) {
                quant::QuantWeight qw;
                qw.type = quant::QuantType::Q4_1;
                qw.in = shape[0];
                qw.out = shape[1];
                qw.bytes = std::move(packed);
                quant_out->emplace(info.name, std::move(qw));
            }
        } else if (info.dtype == GGMLType::Q8_0) {
            const size_t block_count = (numel + 31) / 32;
            std::vector<uint8_t> packed(block_count * 34);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q8_0(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2) {
                quant::QuantWeight qw;
                qw.type = quant::QuantType::Q8_0;
                qw.in = shape[0];
                qw.out = shape[1];
                qw.bytes = std::move(packed);
                quant_out->emplace(info.name, std::move(qw));
            }
        } else if (info.dtype == GGMLType::Q4_K) {
            if (numel % 256 != 0) {
                skip_or_fail(info, "size not divisible by 256");
                continue;
            }
            const size_t sb_count = numel / 256;
            std::vector<uint8_t> packed(sb_count * 144);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q4_k(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2) {
                quant::QuantWeight qw;
                if (shape[0] % 256 == 0) {
                    // Native Q4_K: raw superblocks go straight to the GPU
                    // repack (kind 6) — ~10% fewer bytes than Q4_1 expansion.
                    qw.type = quant::QuantType::Q4_K;
                    qw.in = shape[0];
                    qw.out = shape[1];
                    qw.bytes = std::move(packed);
                } else {
                    q4_k_to_q41(packed.data(), numel, shape[0], shape[1], qw);
                }
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::Q5_K) {
            if (numel % 256 != 0) {
                skip_or_fail(info, "size not divisible by 256");
                continue;
            }
            const size_t sb_count = numel / 256;
            std::vector<uint8_t> packed(sb_count * 176);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q5_k(packed.data(), numel, values.data());
            // Native Q5_K: raw superblocks feed the GEMV kernels directly.
            if (quant_out != nullptr && shape.size() == 2 && shape[0] % 256 == 0) {
                quant::QuantWeight qw;
                qw.type = quant::QuantType::Q5_K;
                qw.in = shape[0];
                qw.out = shape[1];
                qw.bytes = std::move(packed);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::Q6_K) {
            if (numel % 256 != 0) {
                skip_or_fail(info, "size not divisible by 256");
                continue;
            }
            const size_t sb_count = numel / 256;
            std::vector<uint8_t> packed(sb_count * 210);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q6_k(packed.data(), numel, values.data());
            // No native Q6_K GEMV kernel: requantize to Q4_1 so these tensors
            // (attn_v, ffn_down in Q4_K_M mixes) still use the fast path.
            // token_embd is skipped — it only feeds gather + the lm_head
            // fallback, which requantizes from FP32 anyway.
            if (quant_out != nullptr && shape.size() == 2 && info.name != "token_embd.weight") {
                quant::QuantWeight qw;
                quantize_f32_to_q41(values.data(), shape[0], shape[1], qw);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::BF16) {
            std::vector<uint16_t> bf16_values(numel);
            f.read(reinterpret_cast<char *>(bf16_values.data()),
                   static_cast<std::streamsize>(numel * sizeof(uint16_t)));
            for (size_t i = 0; i < numel; ++i) {
                uint32_t bits = static_cast<uint32_t>(bf16_values[i]) << 16;
                float fv;
                std::memcpy(&fv, &bits, sizeof(fv));
                values[i] = fv;
            }
        } else if (info.dtype == GGMLType::Q5_0) {
            if (numel % 32 != 0) {
                skip_or_fail(info, "size not divisible by 32");
                continue;
            }
            const size_t block_count = numel / 32;
            std::vector<uint8_t> packed(block_count * 22);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q5_0(packed.data(), numel, values.data());
            // No native Q5_0 packed kernel: requantize to Q4_1 (same policy as
            // Q6_K above) so these tensors use matvec_q instead of the
            // ~4x-slower FP32 fallback. Skip token_embd — it only feeds
            // gather + the lm_head fallback, which requantizes from FP32 anyway.
            if (quant_out != nullptr && shape.size() == 2 && info.name != "token_embd.weight") {
                quant::QuantWeight qw;
                quantize_f32_to_q41(values.data(), shape[0], shape[1], qw);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::Q5_1) {
            if (numel % 32 != 0) {
                skip_or_fail(info, "size not divisible by 32");
                continue;
            }
            const size_t block_count = numel / 32;
            std::vector<uint8_t> packed(block_count * 24);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q5_1(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2 && info.name != "token_embd.weight") {
                quant::QuantWeight qw;
                quantize_f32_to_q41(values.data(), shape[0], shape[1], qw);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::Q2_K) {
            if (numel % 256 != 0) {
                skip_or_fail(info, "size not divisible by 256");
                continue;
            }
            const size_t sb_count = numel / 256;
            std::vector<uint8_t> packed(sb_count * 84);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q2_k(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2 && info.name != "token_embd.weight") {
                quant::QuantWeight qw;
                quantize_f32_to_q41(values.data(), shape[0], shape[1], qw);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::Q3_K) {
            if (numel % 256 != 0) {
                skip_or_fail(info, "size not divisible by 256");
                continue;
            }
            const size_t sb_count = numel / 256;
            std::vector<uint8_t> packed(sb_count * 110);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q3_k(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2 && info.name != "token_embd.weight") {
                quant::QuantWeight qw;
                quantize_f32_to_q41(values.data(), shape[0], shape[1], qw);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::IQ4_NL) {
            if (numel % 32 != 0) {
                skip_or_fail(info, "size not divisible by 32");
                continue;
            }
            const size_t block_count = numel / 32;
            std::vector<uint8_t> packed(block_count * 18);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_iq4_nl(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2 && info.name != "token_embd.weight") {
                quant::QuantWeight qw;
                quantize_f32_to_q41(values.data(), shape[0], shape[1], qw);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::IQ4_XS) {
            if (numel % 256 != 0) {
                skip_or_fail(info, "size not divisible by 256");
                continue;
            }
            const size_t sb_count = numel / 256;
            std::vector<uint8_t> packed(sb_count * 136);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_iq4_xs(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2 && info.name != "token_embd.weight") {
                quant::QuantWeight qw;
                quantize_f32_to_q41(values.data(), shape[0], shape[1], qw);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::TQ1_0) {
            if (numel % 256 != 0) {
                skip_or_fail(info, "size not divisible by 256");
                continue;
            }
            const size_t sb_count = numel / 256;
            std::vector<uint8_t> packed(sb_count * 54);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_tq1_0(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2 && info.name != "token_embd.weight") {
                quant::QuantWeight qw;
                quantize_f32_to_q41(values.data(), shape[0], shape[1], qw);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::Q8_K) {
            if (numel % 256 != 0) {
                skip_or_fail(info, "size not divisible by 256");
                continue;
            }
            const size_t sb_count = numel / 256;
            std::vector<uint8_t> packed(sb_count * 292);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q8_k(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2 && info.name != "token_embd.weight") {
                quant::QuantWeight qw;
                quantize_f32_to_q41(values.data(), shape[0], shape[1], qw);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else if (info.dtype == GGMLType::TQ2_0) {
            if (numel % 256 != 0) {
                skip_or_fail(info, "size not divisible by 256");
                continue;
            }
            const size_t sb_count = numel / 256;
            std::vector<uint8_t> packed(sb_count * 66);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_tq2_0(packed.data(), numel, values.data());
            if (quant_out != nullptr && shape.size() == 2 && info.name != "token_embd.weight") {
                quant::QuantWeight qw;
                quantize_f32_to_q41(values.data(), shape[0], shape[1], qw);
                if (qw.valid()) {
                    quant_out->emplace(info.name, std::move(qw));
                }
            }
        } else {
            // Unreachable in practice (pre-validation filters unsupported
            // dtypes); kept as a safety net.
            skip_or_fail(info, "uses unsupported dtype " + std::to_string(static_cast<unsigned>(info.dtype)));
            continue;
        }

        core::Tensor tensor(shape, hal::DType::Float32);
        // Freshly allocated contiguous FP32 tensor: bulk copy instead of
        // per-element operator[], which pays a syncToHost() check per weight.
        std::memcpy(tensor.data(), values.data(), values.size() * sizeof(float));
        result.emplace(info.name, std::move(tensor));
    }

    return result;
}

std::string gguf_architecture(const std::string &path)
{
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return {};
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return {};
    }

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type)) {
            break;
        }
        if (key == "general.architecture" && type == 8) {
            std::string arch;
            if (read_string_kv_value(f, &arch)) {
                return arch;
            }
            return {};
        }
        if (!skip_gguf_value(f, type)) {
            break;
        }
    }
    return {};
}

serius::model::ModelConfig gguf_to_model_config(const std::string &path)
{
    serius::model::ModelConfig config;
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return config;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return config;
    }

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type)) {
            break;
        }

        // GGUF arch keys are prefixed per-family (llama.*, qwen2.*, qwen3.*,
        // ...). Match on the suffix after the first dot so every arch works.
        const size_t dot = key.find('.');
        const std::string_view skey =
            (dot == std::string::npos) ? std::string_view(key) : std::string_view(key).substr(dot + 1);

        uint64_t value = 0;
        float fvalue = 0.0f;
        if (skey == "rope.freq_base" && read_float_value(f, type, &fvalue)) {
            config.rope_theta = fvalue;
        } else if (skey == "attention.layer_norm_rms_epsilon" && read_float_value(f, type, &fvalue)) {
            config.rms_norm_eps = fvalue;
        } else if (read_scalar_value(f, type, &value)) {
            if (skey == "vocab_size") {
                config.vocab_size = static_cast<size_t>(value);
            } else if (skey == "embedding_length") {
                config.d_model = static_cast<size_t>(value);
            } else if (skey == "block_count") {
                config.n_layers = static_cast<size_t>(value);
            } else if (skey == "attention.head_count") {
                config.n_heads = static_cast<size_t>(value);
                config.n_kv_heads = static_cast<size_t>(value);
            } else if (skey == "attention.head_count_kv") {
                config.n_kv_heads = static_cast<size_t>(value);
            } else if (skey == "feed_forward_length") {
                config.d_ff = static_cast<size_t>(value);
            } else if (skey == "context_length") {
                config.max_seq_len = static_cast<size_t>(value);
            }
        } else {
            if (!skip_gguf_value(f, type)) {
                break;
            }
        }
    }

    return config;
}

// ---------------------------------------------------------------------------
// GGUF tokenizer extraction
// Reads the tokenizer vocabulary and merges from GGUF metadata KV pairs and
// returns them as a GGUFTokenizerData struct the caller can feed directly to
// BPETokenizer::load_gguf().
// ---------------------------------------------------------------------------
GGUFTokenizerData load_gguf_tokenizer(const std::string &path)
{
    GGUFTokenizerData out;
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return out;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return out;
    }

    // Every array count below is attacker-controlled; cap it and bail on the
    // first failed read so a corrupt file can't spin the loop or OOM the
    // process via reserve()/push_back().
    constexpr uint64_t kMaxTokenizerEntries = 8ull * 1024 * 1024;

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type)) {
            break;
        }

        // ---- tokenizer.ggml.tokens  (type 9 = array of string) ----
        if (key == "tokenizer.ggml.tokens" && type == 9) {
            uint32_t n_elem_type = read_u32_le(f);
            (void)n_elem_type;  // should be 8 (string)
            uint64_t n_tokens = read_u64_le(f);
            if (!f || n_tokens > kMaxTokenizerEntries) {
                break;
            }
            out.tokens.reserve(static_cast<size_t>(n_tokens));
            for (uint64_t t = 0; t < n_tokens; ++t) {
                std::string tok;
                if (!read_string_kv_value(f, &tok)) {
                    break;
                }
                out.tokens.push_back(std::move(tok));
            }
        }
        // ---- tokenizer.ggml.token_type (array of int32) ----
        else if (key == "tokenizer.ggml.token_type" && type == 9) {
            uint32_t n_elem_type = read_u32_le(f);
            (void)n_elem_type;
            uint64_t n_types = read_u64_le(f);
            if (!f || n_types > kMaxTokenizerEntries) {
                break;
            }
            out.token_types.reserve(static_cast<size_t>(n_types));
            for (uint64_t t = 0; t < n_types; ++t) {
                int32_t tt = read_i32_le(f);
                if (!f) {
                    break;
                }
                out.token_types.push_back(tt);
            }
        }
        // ---- tokenizer.ggml.scores (array of float32) ----
        else if (key == "tokenizer.ggml.scores" && type == 9) {
            uint32_t n_elem_type = read_u32_le(f);
            (void)n_elem_type;
            uint64_t n_scores = read_u64_le(f);
            if (!f || n_scores > kMaxTokenizerEntries) {
                break;
            }
            out.scores.reserve(static_cast<size_t>(n_scores));
            for (uint64_t t = 0; t < n_scores; ++t) {
                float sc = 0.0f;
                f.read(reinterpret_cast<char *>(&sc), sizeof(sc));
                if (!f) {
                    break;
                }
                out.scores.push_back(sc);
            }
        }
        // ---- tokenizer.ggml.merges (array of string) ----
        else if (key == "tokenizer.ggml.merges" && type == 9) {
            uint32_t n_elem_type = read_u32_le(f);
            (void)n_elem_type;
            uint64_t n_merges = read_u64_le(f);
            if (!f || n_merges > kMaxTokenizerEntries) {
                break;
            }
            out.merges.reserve(static_cast<size_t>(n_merges));
            for (uint64_t m = 0; m < n_merges; ++m) {
                std::string merge;
                if (!read_string_kv_value(f, &merge)) {
                    break;
                }
                out.merges.push_back(std::move(merge));
            }
        }
        // ---- tokenizer.ggml.bos_token_id ----
        else if (key == "tokenizer.ggml.bos_token_id") {
            uint64_t v = 0;
            if (read_scalar_value(f, type, &v))
                out.bos_token_id = static_cast<int32_t>(v);
            else
                skip_gguf_value(f, type);
        }
        // ---- tokenizer.ggml.eos_token_id ----
        else if (key == "tokenizer.ggml.eos_token_id") {
            uint64_t v = 0;
            if (read_scalar_value(f, type, &v))
                out.eos_token_id = static_cast<int32_t>(v);
            else
                skip_gguf_value(f, type);
        }
        // ---- tokenizer.ggml.model (string: "llama", "gpt2", …) ----
        else if (key == "tokenizer.ggml.model" && type == 8) {
            read_string_kv_value(f, &out.tokenizer_model);
        } else {
            if (!skip_gguf_value(f, type)) {
                break;
            }
        }
    }

    out.valid = !out.tokens.empty();
    return out;
}

}  // namespace io
}  // namespace serius
