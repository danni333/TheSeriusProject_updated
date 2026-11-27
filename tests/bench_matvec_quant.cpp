// Microbenchmark for core::blas::matvec_q (ggml-style quantized GEMV).
// Measures per-call latency and effective weight-stream bandwidth for each
// supported QuantType at a Llama-7B-like decode shape (K=4096, N=11008).
//
// Run from build/: ./tests/Release/bench_matvec_quant.exe
// Env knobs (read by matvec_q): SERIUS_NO_INTDOT, SERIUS_INTDOT,
// SERIUS_NO_VNNI, SERIUS_NO_AVX512, OMP_NUM_THREADS.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "blas/blas.h"
#include "quant/quant_weight.h"

using namespace serius;

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static uint32_t rng_state = 0x12345678u;
static uint32_t rnd()
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

// Fill every block's fp16 scale fields with sane constants:
//   d = 1.0 (0x3C00), m/dmin = 0.25 (0x3400). Quant payloads stay random.
static void fix_scales(quant::QuantWeight &w)
{
    const size_t in = w.in, out = w.out;
    auto *p = w.bytes.data();
    auto put16 = [](uint8_t *q, uint16_t v) {
        q[0] = static_cast<uint8_t>(v & 0xFF);
        q[1] = static_cast<uint8_t>(v >> 8);
    };
    if (w.type == quant::QuantType::Q4_K || w.type == quant::QuantType::Q5_K) {
        const size_t sb = (w.type == quant::QuantType::Q4_K) ? 144 : 176;
        for (size_t r = 0; r < out; ++r) {
            for (size_t s = 0; s < in / 256; ++s) {
                uint8_t *b = p + (r * (in / 256) + s) * sb;
                put16(b, 0x3C00);
                put16(b + 2, 0x3400);
            }
        }
        return;
    }
    const size_t bb = quant::QuantWeight::block_bytes(w.type);
    const size_t bpr = (in + 31) / 32;
    for (size_t r = 0; r < out; ++r) {
        for (size_t b = 0; b < bpr; ++b) {
            uint8_t *blk = p + (r * bpr + b) * bb;
            put16(blk, 0x3C00);
            if (w.type == quant::QuantType::Q4_1) {
                put16(blk + 2, 0x3400);
            }
        }
    }
}

static quant::QuantWeight make_weight(quant::QuantType t, size_t in, size_t out)
{
    quant::QuantWeight w;
    w.type = t;
    w.in = in;
    w.out = out;
    w.bytes.resize(out * w.row_bytes());
    for (auto &b : w.bytes) {
        b = static_cast<uint8_t>(rnd());
    }
    fix_scales(w);
    return w;
}

static void bench_one(const char *name, quant::QuantType t, size_t in, size_t out)
{
    quant::QuantWeight w = make_weight(t, in, out);
    std::vector<float> x(in), y(out);
    for (auto &v : x) {
        v = static_cast<float>(static_cast<int>(rnd() % 2001) - 1000) / 1000.0f;
    }

    // Warmup: touch every page + warm TLB/branch predictors.
    for (int i = 0; i < 2; ++i) {
        core::blas::matvec_q(w, x.data(), y.data());
    }

    const int iters = 31;
    std::vector<double> ts;
    ts.reserve(iters);
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_ms();
        core::blas::matvec_q(w, x.data(), y.data());
        const double dt = now_ms() - t0;
        ts.push_back(dt);
        if (dt < best) {
            best = dt;
        }
    }
    std::sort(ts.begin(), ts.end());
    const double med = ts[ts.size() / 2];
    const double bytes = static_cast<double>(w.bytes.size());
    std::printf("%-6s K=%zu N=%-6zu  best %8.3f ms  med %8.3f ms  %7.1f GB/s(med)  (%6.1f MB)\n",
                name, in, out, best, med, bytes / med / 1.0e6, bytes / 1048576.0);
}

int main()
{
    std::printf("matvec_q microbenchmark — %s\n\n", "serius");

    // DRAM ceiling: streaming read bandwidth (read-only pass over a buffer
    // larger than L3). Volatile sink prevents elision.
    {
        const size_t mb = 256;
        std::vector<uint8_t> buf(mb * 1024 * 1024);
        for (auto &b : buf) b = static_cast<uint8_t>(rnd());
        volatile uint8_t sink = 0;
        for (int i = 0; i < 3; ++i) sink ^= buf[i];
        const size_t chunk = (buf.size() + 255) / 256;
        double best = 1e30;
        for (int i = 0; i < 15; ++i) {
            const double t0 = now_ms();
#pragma omp parallel for schedule(static) reduction(^ : sink)
            for (int64_t c = 0; c < 256; ++c) {
                const size_t off0 = static_cast<size_t>(c) * chunk;
                const size_t off1 = std::min(buf.size(), off0 + chunk);
                uint64_t acc = 0;
                for (size_t off = off0; off + 8 <= off1; off += 8) {
                    uint64_t v;
                    std::memcpy(&v, buf.data() + off, 8);
                    acc += v;
                }
                sink ^= static_cast<uint8_t>(acc);
            }
            best = std::min(best, now_ms() - t0);
        }
        std::printf("memcpy-read ceiling: %7.1f GB/s (%zu MB)\n\n", buf.size() / best / 1.0e6, mb);
    }
    const size_t K = 4096, N = 11008;
    bench_one("Q4_0", quant::QuantType::Q4_0, K, N);
    bench_one("Q4_1", quant::QuantType::Q4_1, K, N);
    bench_one("Q8_0", quant::QuantType::Q8_0, K, N);
    bench_one("Q4_K", quant::QuantType::Q4_K, K, N);
    bench_one("Q5_K", quant::QuantType::Q5_K, K, N);

    // Attention-projection-like shape (square-ish, K=hidden).
    const size_t K2 = 4096, N2 = 4096;
    bench_one("Q4_0", quant::QuantType::Q4_0, K2, N2);
    bench_one("Q8_0", quant::QuantType::Q8_0, K2, N2);
    bench_one("Q4_K", quant::QuantType::Q4_K, K2, N2);
    bench_one("Q5_K", quant::QuantType::Q5_K, K2, N2);
    return 0;
}
