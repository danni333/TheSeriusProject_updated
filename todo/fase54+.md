# Fase 54+ — Cose mancanti (Performance benchmarks)

Riferimento: `fase54_performance_benchmarks.md`

---

## 1. Benchmark performance

| Elemento | Stato | Nota |
|----------|--------|-----|
| Benchmark matmul (GFLOPS per size) | Parziale | test_blas possibile; target dedicato da verificare |
| Benchmark inference (tok/s, latency p50/p99) | ❌ | |
| Benchmark training (samples/s, GPU util) | ❌ | |
| Benchmark memoria (pool vs malloc, KV cache) | Parziale | test_memory |
| Report e grafici (script) | ❌ | |
| Confronto con baseline (OpenBLAS, cuBLAS) | ❌ | |

---

**Priorità:** definire suite bench e documentare risultati; allineare a fase25.
