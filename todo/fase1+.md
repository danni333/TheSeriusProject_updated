# Fase 1+ — Cose mancanti (Motore matematico)

Riferimento: `fase1_motore_matematico.md`

---

## 1. Classe Tensor

| Elemento | Stato | Nota |
|----------|--------|-----|
| `data_`, `shape_`, `strides_`, `size_`, `owns_data_` | Da verificare | Controllare in `tensor.h` che i campi e il layout row-major ci siano |
| `reshape(new_shape)` senza copia se compatibile | Da verificare | |
| `slice(dim, start, end)` → view | Da verificare | |
| `fill(value)`, `copy()` | Da verificare | |
| Integrazione Tensor + MemoryPool (costruttore con pool opzionale) | Da verificare | Fase1 richiede allocazione da pool se fornito |
| Test: view vs copy, accesso indicizzato, reshape | ✅ | `test_tensor.cpp` presente |

---

## 2. BLAS / Algebra lineare

| Operazione | Stato | Nota |
|------------|--------|-----|
| `matmul` | ✅ | dispatch, scalar, avx2, avx512, cuda |
| `matvec` | ❓ | Non evidente in blas.h — da implementare se assente |
| `element_wise_add`, `element_wise_mul` | ✅ | elementwise.cpp |
| `transpose` | ❓ | Verificare se in blas o altrove |
| `scale` (α·A) | ❓ | Verificare in elementwise o blas |
| Tiling per cache (blocchi 64–128) | Da verificare | Fase1 richiede tiling per matrici grandi |
| Soglie SIMD (small/medium/large) | Parziale | AUDIT consiglia soglie in dispatch; verificare se applicate |
| Fallback scalare con `__builtin_cpu_supports("avx2")` | ✅ | Dispatch presente |
| `matmul_avx2.cpp`, `matmul_avx512.cpp`, `matmul_scalar.cpp` | ✅ | Presenti |
| Benchmark `bench_matmul` (256, 512, 1024) | ❓ | Verificare se in test_blas o target dedicato |

---

## 3. Memory pool

| Elemento | Stato | Nota |
|----------|--------|-----|
| `MemoryPool` (arena, offset, allocate, reset) | ✅ | memory_pool.h/cpp |
| Allineamento 64 byte | Da verificare | |
| `PoolAllocator<T>` STL-compatible | ✅ | pool_allocator.h |
| `ScopedArena` RAII | Da verificare | |
| `grow()` / blocchi multipli | Da verificare | |
| Integrazione Tensor ↔ pool | Da verificare | |
| Thread-local pool (futuro) | ❌ | AUDIT lo consiglia; non richiesto in Fase1 |
| Test: allocazione, alignment, reset, crescita | ✅ | test_memory.cpp |

---

## 4. Struttura file Fase 1

| File previsto | Stato |
|---------------|--------|
| `src/tensor/tensor.h`, `.cpp` | ✅ |
| `src/blas/blas.h` | ✅ |
| `src/blas/matmul_avx2.cpp`, avx512, scalar | ✅ |
| `src/blas/elementwise.cpp` | ✅ |
| `src/blas/dispatch.cpp` | ✅ |
| `src/memory/memory_pool.h`, `.cpp` | ✅ |
| `src/memory/pool_allocator.h` | ✅ |
| `tests/test_tensor.cpp`, test_blas, test_memory | ✅ |

---

## 5. Piano di verifica

| Test / benchmark | Stato |
|------------------|--------|
| test_tensor (creazione, reshape, slice, view/copy) | ✅ |
| test_blas (correttezza matmul, elementwise) | ✅ |
| test_memory (pool, alignment, reset) | ✅ |
| bench_matmul (256, 512, 1024) | Da aggiungere/verificare |
| bench_memory (throughput pool vs malloc) | Da aggiungere/verificare |

---

**Priorità:** implementare `matvec` e `transpose` se assenti; calibrare soglie SIMD in dispatch; aggiungere benchmark matmul/memory; verificare tiling e integrazione Tensor–pool.
