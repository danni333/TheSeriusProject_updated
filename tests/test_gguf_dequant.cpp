#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "io/gguf_loader.h"

using namespace serius;

namespace {

// fp16 -> fp32, standalone copy so the test does not depend on loader internals.
float h2f(uint16_t h)
{
    const uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 0x1u;
    const uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    const uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;
    if (exp == 0u && mant != 0u) {
        const float v = static_cast<float>(mant) * 0x1p-24f;
        return sign ? -v : v;
    }
    uint32_t bits;
    if (exp == 0u) {
        bits = sign << 31;
    } else if (exp == 0x1Fu) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// ---------------------------------------------------------------------------
// Independent reference dequantizers, transcribed from ggml
// (ggml-quants.c dequantize_row_q2_K / dequantize_row_q3_K). If the loader
// implementation drifts from the ggml block layout these comparisons fail.
// ---------------------------------------------------------------------------

void ref_q2_k(const uint8_t *x, float *y, size_t n)
{
    const size_t nb = n / 256;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = x + i * 84;
        const float d = h2f(static_cast<uint16_t>(blk[80] | (blk[81] << 8)));
        const float min = h2f(static_cast<uint16_t>(blk[82] | (blk[83] << 8)));
        const uint8_t *q = blk + 16;
        const uint8_t *scales = blk;
        int is = 0;
        for (int nn = 0; nn < 256; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                const float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<int8_t>((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                const float dl2 = d * (sc & 0xF), ml2 = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl2 * static_cast<int8_t>((q[l + 16] >> shift) & 3) - ml2;
                shift += 2;
            }
            q += 32;
        }
    }
}

void ref_q3_k(const uint8_t *x, float *y, size_t n)
{
    const size_t nb = n / 256;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t *blk = x + i * 110;
        const float d_all = h2f(static_cast<uint16_t>(blk[108] | (blk[109] << 8)));
        const uint8_t *q = blk + 32;
        const uint8_t *hm = blk;
        const uint8_t *scales12 = blk + 96;

        uint32_t aux[4];
        std::memcpy(aux, scales12, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = reinterpret_cast<const int8_t *>(aux);

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

// ---------------------------------------------------------------------------
// Minimal GGUF writer: header + tensor infos + 32-byte-aligned payload.
// Only what load_all_tensors needs — no KV metadata.
// ---------------------------------------------------------------------------

struct TestTensor
{
    std::string name;
    uint32_t dtype;  // GGMLType value
    std::vector<uint64_t> dims;  // GGUF order, as stored on disk
    std::vector<uint8_t> payload;
};

void put_u32(std::ofstream &f, uint32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); }
void put_u64(std::ofstream &f, uint64_t v) { f.write(reinterpret_cast<const char *>(&v), 8); }

void write_min_gguf(const std::filesystem::path &path, const std::vector<TestTensor> &tensors)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(f.good());
    put_u32(f, 0x46554747u);  // "GGUF"
    put_u32(f, 3);            // version
    put_u64(f, tensors.size());
    put_u64(f, 0);  // n_kv

    std::vector<uint64_t> offsets(tensors.size());
    // Tensor infos are written first; data section starts at the next
    // 32-byte boundary relative to the file start (loader alignment rule).
    uint64_t info_end = 24;
    for (const auto &t : tensors) {
        info_end += 8 + t.name.size() + 4 + 8 * t.dims.size() + 4 + 8;
    }
    const uint64_t data_base = (info_end + 31) & ~uint64_t(31);
    uint64_t off = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        offsets[i] = off;
        off += tensors[i].payload.size();
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &t = tensors[i];
        put_u64(f, t.name.size());
        f.write(t.name.data(), static_cast<std::streamsize>(t.name.size()));
        put_u32(f, static_cast<uint32_t>(t.dims.size()));
        for (uint64_t d : t.dims) put_u64(f, d);
        put_u32(f, t.dtype);
        put_u64(f, offsets[i]);
    }
    const uint64_t pad = data_base - info_end;
    for (uint64_t i = 0; i < pad; ++i) f.put('\0');
    for (const auto &t : tensors) {
        f.write(reinterpret_cast<const char *>(t.payload.data()), static_cast<std::streamsize>(t.payload.size()));
    }
}

uint16_t fp16_of(float v)
{
    // Only used for a handful of exactly-representable constants.
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = static_cast<int>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

constexpr uint16_t F16_ONE = 0x3C00;    // 1.0
constexpr uint16_t F16_HALF = 0x3800;   // 0.5

// Build `nblocks` Q2_K superblocks (84B each) with pseudo-random content.
std::vector<uint8_t> make_q2k_blocks(size_t nblocks, std::mt19937 &rng)
{
    std::vector<uint8_t> out(nblocks * 84);
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t b = 0; b < nblocks; ++b) {
        uint8_t *blk = out.data() + b * 84;
        for (int i = 0; i < 80; ++i) blk[i] = static_cast<uint8_t>(byte(rng));
        const uint16_t d = F16_ONE, dmin = F16_HALF;
        blk[80] = static_cast<uint8_t>(d & 0xFF);
        blk[81] = static_cast<uint8_t>(d >> 8);
        blk[82] = static_cast<uint8_t>(dmin & 0xFF);
        blk[83] = static_cast<uint8_t>(dmin >> 8);
    }
    return out;
}

// Build `nblocks` Q3_K superblocks (110B each) with pseudo-random content.
std::vector<uint8_t> make_q3k_blocks(size_t nblocks, std::mt19937 &rng)
{
    std::vector<uint8_t> out(nblocks * 110);
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t b = 0; b < nblocks; ++b) {
        uint8_t *blk = out.data() + b * 110;
        for (int i = 0; i < 108; ++i) blk[i] = static_cast<uint8_t>(byte(rng));
        const uint16_t d = F16_ONE;
        blk[108] = static_cast<uint8_t>(d & 0xFF);
        blk[109] = static_cast<uint8_t>(d >> 8);
    }
    return out;
}

void expect_close(const core::Tensor &t, const std::vector<float> &ref, const char *what)
{
    ASSERT_EQ(t.size(), ref.size()) << what;
    const float *got = t.data();
    for (size_t i = 0; i < ref.size(); ++i) {
        ASSERT_NEAR(got[i], ref[i], 1e-5f) << what << " mismatch at element " << i;
    }
}

}  // namespace

TEST(GGUFDequantTest, Q2KMatchesGgmlReference)
{
    std::mt19937 rng(1234);
    const auto path = std::filesystem::temp_directory_path() / "serius_test_q2k.gguf";
    std::filesystem::remove(path);

    // 1024 weights = 4 superblocks, stored as GGUF dims {256, 4}.
    auto payload = make_q2k_blocks(4, rng);
    std::vector<TestTensor> tensors = {
        {"test_q2k.weight", static_cast<uint32_t>(io::GGMLType::Q2_K), {256, 4}, std::move(payload)},
    };
    write_min_gguf(path, tensors);

    std::unordered_map<std::string, quant::QuantWeight> qmap;
    auto loaded = io::load_all_tensors(path.string(), &qmap);
    std::filesystem::remove(path);

    auto it = loaded.find("test_q2k.weight");
    ASSERT_NE(it, loaded.end());
    EXPECT_EQ(it->second.shape(), (std::vector<size_t>{256, 4}));

    std::vector<float> ref(1024);
    ref_q2_k(tensors[0].payload.data(), ref.data(), ref.size());
    expect_close(it->second, ref, "Q2_K");

    // 2-D non-embedding tensors must also produce a packed Q4_1 QuantWeight.
    auto qit = qmap.find("test_q2k.weight");
    ASSERT_NE(qit, qmap.end());
    EXPECT_EQ(qit->second.type, quant::QuantType::Q4_1);
    EXPECT_EQ(qit->second.in, 256u);
    EXPECT_EQ(qit->second.out, 4u);
    EXPECT_TRUE(qit->second.valid());
}

TEST(GGUFDequantTest, Q3KMatchesGgmlReference)
{
    std::mt19937 rng(5678);
    const auto path = std::filesystem::temp_directory_path() / "serius_test_q3k.gguf";
    std::filesystem::remove(path);

    // Regression for the superblock layout fix: {hmask[32], qs[64],
    // scales[12], fp16 d} = 110 bytes, d at offset 108.
    auto payload = make_q3k_blocks(4, rng);
    std::vector<TestTensor> tensors = {
        {"blk.0.ffn_down.weight", static_cast<uint32_t>(io::GGMLType::Q3_K), {256, 4}, std::move(payload)},
    };
    write_min_gguf(path, tensors);

    auto loaded = io::load_all_tensors(path.string(), nullptr);
    std::filesystem::remove(path);

    auto it = loaded.find("blk.0.ffn_down.weight");
    ASSERT_NE(it, loaded.end());
    EXPECT_EQ(it->second.shape(), (std::vector<size_t>{256, 4}));

    std::vector<float> ref(1024);
    ref_q3_k(tensors[0].payload.data(), ref.data(), ref.size());
    expect_close(it->second, ref, "Q3_K");
}

TEST(GGUFDequantTest, MixedQuantFileKeepsTensorOffsets)
{
    std::mt19937 rng(99);
    const auto path = std::filesystem::temp_directory_path() / "serius_test_mixed.gguf";
    std::filesystem::remove(path);

    auto q2k = make_q2k_blocks(2, rng);
    auto q3k = make_q3k_blocks(2, rng);
    std::vector<float> f32vals = {1.5f, -2.25f, 0.0f, 3.0f};
    std::vector<uint8_t> f32bytes(4 * sizeof(float));
    std::memcpy(f32bytes.data(), f32vals.data(), f32bytes.size());

    std::vector<TestTensor> tensors = {
        {"alpha_q2k.weight", static_cast<uint32_t>(io::GGMLType::Q2_K), {256, 2}, std::move(q2k)},
        {"beta_f32.weight", static_cast<uint32_t>(io::GGMLType::F32), {2, 2}, std::move(f32bytes)},
        {"gamma_q3k.weight", static_cast<uint32_t>(io::GGMLType::Q3_K), {256, 2}, std::move(q3k)},
    };
    write_min_gguf(path, tensors);

    auto loaded = io::load_all_tensors(path.string(), nullptr);
    std::filesystem::remove(path);

    ASSERT_EQ(loaded.size(), 3u);

    auto fit = loaded.find("beta_f32.weight");
    ASSERT_NE(fit, loaded.end());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(fit->second.data()[i], f32vals[i]);
    }

    std::vector<float> ref2k(512), ref3k(512);
    ref_q2_k(tensors[0].payload.data(), ref2k.data(), ref2k.size());
    ref_q3_k(tensors[2].payload.data(), ref3k.data(), ref3k.size());
    expect_close(loaded.at("alpha_q2k.weight"), ref2k, "mixed Q2_K");
    expect_close(loaded.at("gamma_q3k.weight"), ref3k, "mixed Q3_K");
}
