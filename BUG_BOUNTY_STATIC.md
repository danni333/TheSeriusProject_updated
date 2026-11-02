# Static Analysis Report (Fase 0-1 parziale)

## Findings

### S1 — Debug prints in production code
- `src/blas/matmul_avx2.cpp:49` printf DEBUG.
- `src/infer/llama_inference.cpp:701-831` multiple printf DEBUG con fflush.

Impatto: performance e rumore log. Fix: rimuovere o guarded da macro DEBUG.

### S2 — Backend stub non compilabili (Metal/Vulkan)
- `src/hal/metal_backend.cpp`, `src/hal/vulkan_backend.cpp`: mancano `malloc_pinned`, `malloc_managed`, `free_pinned` di `IBackend`; classi astratte. `is_available()` false.
- TODOs: 6 totali.

Impatto: abilitare backend causa errore compilazione. Fix: implementare metodi o rimuovere backend dall'interfaccia.

### S3 — TODO count basso (6)
Solo HAL. Altri gap non documentati (vedi MISSING_PARTS_REPORT).

### S4 — Nessun unsafe strcpy/gets/sprintf
Gli `fgets` usano buffer bounded.

### S5 — Possibili divisioni per zero
Nessun pattern `/ 0` evidente; verificare runtime con input shape 0.

## Next static steps
- Analizzare ogni modulo per unchecked vector access, null deref, integer overflow.
- Aggiungere macro SERIUS_CHECK in punti critici.
- Rimuovere debug prints.
- Documentare stub HAL.
