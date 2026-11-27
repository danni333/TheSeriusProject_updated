// Microbenchmark: costo di una regione `#pragma omp parallel for`.
// Decisione Fase 73e (loop persistente stile ggml): se il wake-up OpenMP
// costa poco, il progetto non conviene. Vedi todo/fase73e_ggml_persistent_loop.md.
//
// Misura: R regioni per round, round ripetuti → distribuzione di µs/region
// (best/median/p90). Body minimale (somma volatile) per non essere
// ottimizzato via, ma abbastanza piccolo da misurare quasi solo il dispatch.
//
// Run da build/: ./tests/Release/bench_omp_region.exe
// Override: OMP_NUM_THREADS non serve — il bench fissa 8 e 16 via omp_set_num_threads.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_us()
{
    return std::chrono::duration<double, std::micro>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

// Body "reale ma minuscolo": ogni thread tocca una variabile volatile
// per impedire la cancellazione della regione da parte del compilatore.
static volatile int g_sink;

static void report(const char *label, int nthreads, int inner_mult)
{
    constexpr int kRegions = 500;   // regioni per round
    constexpr int kRounds = 21;     // round per misura

    std::vector<double> samples;
    samples.reserve(kRounds);
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
    for (int w = 0; w < 200; ++w) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nthreads * inner_mult; ++i) {
            g_sink += i & 1;
        }
    }
    for (int r = 0; r < kRounds; ++r) {
        const double t0 = now_us();
        for (int k = 0; k < kRegions; ++k) {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < nthreads * inner_mult; ++i) {
                g_sink += i & 1;
            }
        }
        samples.push_back((now_us() - t0) / kRegions);
    }
    std::sort(samples.begin(), samples.end());

    // Baseline seriale: stesso numero di iterazioni totali.
    const int total_iters = kRounds * kRegions * nthreads * inner_mult;
    double t0 = now_us();
    for (int r = 0; r < total_iters; ++r) {
        g_sink += r & 1;
    }
    const double serial_us = (now_us() - t0) / (kRounds * kRegions);

    std::printf("%-30s threads=%2d  best=%7.2f us  median=%7.2f us  p90=%7.2f us  (serial-equiv=%6.3f us)\n",
                label, nthreads, samples.front(), samples[samples.size() / 2],
                samples[(samples.size() * 9) / 10], serial_us);
}

int main()
{
#ifdef _OPENMP
    std::printf("OpenMP runtime: max_threads=%d\n", omp_get_max_threads());
#else
    std::printf("WARNING: OpenMP non disponibile — #pragma ignorato, numeri=seriali\n");
#endif
    std::printf("Costo dispatch di una regione 'omp parallel for' (best/median/p90, us/region)\n\n");

    for (int nt : {4, 8, 12, 16}) {
        report("empty-ish (4 it totali)", nt, 4);
    }

    // Scenario più realistico: ~64 iterazioni per thread — una fetta
    // piccola di lavoro come in un GEMV chunk.
    for (int nt : {8, 16}) {
        report("64 it/thread", nt, 64);
    }

    return 0;
}
