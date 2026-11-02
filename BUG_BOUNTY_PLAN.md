# Piano Bug Bounty — TheSeriusProject (copertura totale)

Obiettivo: coprire l'intera repository in modo sistematico e con calma, identificando bug, regressioni, vulnerabilità e gap di implementazione, senza refactor ampi. Ogni fase produce evidenza (log, test, riproduzione) e fix mirati.

## Principi
- Calma: una fase per volta, nessun grande refactor.
- Diagnostica prima: aggiungere log/output per validare ipotesi prima di modificare.
- Fix minimi e mirati.
- Confermare la diagnosi prima di applicare il fix.
- Ogni bug tracciato con ID, file:linea, severità, riproduzione.

## Scope
- Codice C++ (src/, 404 file, 43 moduli)
- Test (tests/) e build CMake
- Backend HAL (CUDA/Metal/Vulkan/CPU)
- Distributed (MPI/NCCL), quantizzazione, autograd, transformer, tokenizer, server, CLI, hub/plugin
- Python scripts (fix_elementwise.py)
- Config e docs

## Fasi

### Fase 0 — Setup e baseline (1-2 gg)
- Build pulita Debug+Release, catturare errori/warning.
- Eseguire suite test esistente, salvare risultati.
- Inventario moduli e dipendenze.
- Definire severità: Critical/High/Medium/Low.

### Fase 1 — Build e compilazione (2-3 gg)
- Risolvere warning/errori GCC/Clang.
- Verificare opzioni CMake (CUDA, MPI, NCCL, etc.).
- Controllare header include guard, dipendenze circolari.
- Output: build pulita senza warning (o lista accettata).

### Fase 2 — Core tensor/BLAS/autograd (3-5 gg)
- Test numerici (NaN/Inf, shape mismatch, aliasing).
- Edge case: tensor vuoti, broadcast, stride.
- Autograd: grafo, gradienti, memoria.
- Aggiungere test unitari mirati.

### Fase 3 — Transformer/attention/KV (3-5 gg)
- Flash attention, RoPE, causal mask.
- KV cache paging, eviction.
- Precisione numerica vs riferimento.
- Test determinismo.

### Fase 4 — Quantizzazione (2-4 gg)
- Int8/Int4/fp16, dequant, round-trip.
- Per-channel, group, outlier.
- Confronto accuratezza.

### Fase 5 — Tokenizer/IO/model loading (2-3 gg)
- Parsing JSON/Safetensors/GGUF.
- Edge case vocab, unicode, BPE.
- Corruzione file handling.

### Fase 6 — Backend HAL (2-4 gg)
- CPU SIMD paths vs scalar.
- CUDA kernels (se disponibile).
- Metal/Vulkan stubs: compilabilità (abstract methods).
- Memory pool, pinned/managed.

### Fase 7 — Distributed (3-5 gg)
- MPI/NCCL init, collectives.
- DP/PP/TP correctness.
- Fault tolerance, timeout.
- Multi-process test harness.

### Fase 8 — Training/optim/finetune/LoRA (3-4 gg)
- Loss, optimizer step, checkpoint save/load.
- Gradient accumulation, mixed precision.
- LoRA merge/unmerge.

### Fase 9 — Inference/server/CLI (2-3 gg)
- API OpenAI compat, streaming SSE.
- Concurrency, backpressure.
- Validazione input, auth.

### Fase 10 — Agent/RAG/tools/MCP (2-4 gg)
- Action parser, planner, react.
- Tool calling, sandbox plugin.
- Sicurezza: injection, path traversal.

### Fase 11 — Multimodal/audio/vision (2-3 gg)
- Encoder/decoder, preprocessing.
- Shape/batch handling.

### Fase 12 — Memory/cache/hub (2-3 gg)
- Paged KV, prefix cache.
- Model hub download, integrity.

### Fase 13 — Stabilità/stress/fuzz (3-5 gg)
- Stress test, sanitizers (ASan/UBSan/TSan).
- Fuzzing input parser.
- Memory leak detection.

### Fase 14 — Sicurezza e hardening (2-3 gg)
- Path traversal, deserialization.
- Resource limits, DoS.
- Secret handling.

### Fase 15 — Documentazione/UX (1-2 gg)
- README, CLI help, error messages.
- Coerenza docs.

### Fase 16 — Regression e CI (2-3 gg)
- Test matrix, CI pipeline.
- Coverage report.
- Release checklist.

## Cadence
- Settimana 1-2: Fasi 0-3
- Settimana 3-4: Fasi 4-7
- Settimana 5-6: Fasi 8-11
- Settimana 7-8: Fasi 12-16

## Deliverable per fase
- Report bug (markdown) con ID, riproduzione, fix.
- Patch minime + test.
- Aggiornamento CHANGELOG.

## Tracking
Usare todo/ o nuova cartella bugbounty/ con file per fase: bugbounty/phaseXX.md.

## Note
- Non modificare API senza necessità.
- Mantenere compatibilità GCC 15 (già in corso).
- Priorità ai bug che causano crash/corruzione dati.
