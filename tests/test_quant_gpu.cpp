// GPU-side verification of the packed-quant GEMV path: repack a synthetic
// Q5_K weight (dev_kind 7), run gemv_quant_vec_f32 on the CUDA backend, and
// compare against the CPU matvec_q reference.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "blas/blas.h"
#include "hal/device.h"
#include "hal/fused_kernels.h"
#include "quant/quant_weight.h"

using namespace serius;

namespace {

uint32_t g_rng = 0xC0FFEEu;
uint32_t rnd()
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng >> 8;
}

quant::QuantWeight make_q5k(size_t in, size_t out)
{
    quant::QuantWeight w;
    w.type = quant::QuantType::Q5_K;
    w.in = in;
    w.out = out;
    w.bytes.resize(out * w.row_bytes());
    for (auto &b : w.bytes) {
        b = static_cast<uint8_t>(rnd());
    }
    // fp16 d = 1.0 (0x3C00), dmin = 0.25 (0x3400) in every superblock.
    for (size_t r = 0; r < out; ++r) {
        for (size_t s = 0; s < in / 256; ++s) {
            uint8_t *blk = w.bytes.data() + (r * (in / 256) + s) * 176;
            blk[0] = 0x00; blk[1] = 0x3C;
            blk[2] = 0x00; blk[3] = 0x34;
        }
    }
    return w;
}

}  // namespace

TEST(QuantGpu, Q5KRepackMatchesCpu)
{
#ifdef SERIUS_USE_CUDA
    hal::IBackend *be = hal::get_backend(hal::DeviceType::CUDA);
    if (be == nullptr || !be->is_available()) {
        GTEST_SKIP() << "CUDA backend unavailable";
    }

    const size_t in = 512, out = 37;  // 2 superblocks/row, odd row count
    quant::QuantWeight w = make_q5k(in, out);
    std::vector<float> x(in), y_cpu(out, -1.0f);
    for (auto &v : x) {
        v = static_cast<float>(static_cast<int>(rnd() % 2001) - 1000) / 1000.0f;
    }
    core::blas::matvec_q(w, x.data(), y_cpu.data());

    quant::repack_quant_for_gpu(w);
    ASSERT_EQ(w.dev_kind, 7);
    ASSERT_NE(w.dev_stride, 0u);

    void *dx = be->malloc(in * sizeof(float));
    void *dw = be->malloc(w.dev_bytes.size());
    void *dy = be->malloc(out * sizeof(float));
    ASSERT_NE(dx, nullptr);
    ASSERT_NE(dw, nullptr);
    ASSERT_NE(dy, nullptr);
    be->memcpy_h2d(dx, x.data(), in * sizeof(float));
    be->memcpy_h2d(dw, w.dev_bytes.data(), w.dev_bytes.size());

    hal::gemv_quant_vec_f32(dx, nullptr, dw, w.dev_kind, w.in, w.out, static_cast<int>(w.dev_stride),
                            static_cast<int>(w.dev_qs), static_cast<int>(w.dev_d), static_cast<int>(w.dev_m),
                            dy, nullptr);

    std::vector<float> y_gpu(out, 0.0f);
    be->memcpy_d2h(y_gpu.data(), dy, out * sizeof(float));

    for (size_t r = 0; r < out; ++r) {
        const float tol = std::abs(y_cpu[r]) * 1e-3f + 1e-2f;
        EXPECT_NEAR(y_gpu[r], y_cpu[r], tol) << "row " << r;
    }

    be->free(dx);
    be->free(dw);
    be->free(dy);
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}
