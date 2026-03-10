# Fase 25+ — Cose mancanti (Benchmarking)

Riferimento: `fase25_benchmarking.md`

---

## 1. Benchmark

| Elemento | Stato | Nota |
|----------|--------|-----|
| Benchmark matmul/memory (fase1) | Parziale | Verificare target bench_* in CMake |
| Benchmark inference (throughput, latency) | ❌ | |
| Benchmark training (samples/sec, GPU util) | ❌ | |
| Suite benchmark standard (es. MLPerf-style) | ❌ | |
| Script benchmark.sh / report | ❌ | scripts/ |
| configs/ per benchmark | ❌ | |

---

**Priorità:** definire target ctest o eseguibili bench_* e script di report.
