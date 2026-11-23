# Roadmap "prodotto vendibile" — 2026-11-23

Stato corrente e lavoro rimanente per portare TheSeriusProject a prodotto
reale. Documento di handoff: qualunque agente (o sessione futura) riprende
da qui. Baseline misurata in `BENCHMARKS.md` (confronto diretto vs llama.cpp).

## Stato attuale (verificato)

- Decode GPU pienamente competitiva: parità o meglio di llama.cpp sui modelli
  testati (Qwen2.5-0.5B Q4_K_M 469.7 vs 368.7 tok/s; 1.5B 91.7 vs 91.3;
  SmolLM2-135M Q4_0 653 vs 567). CUDA graph + pesi packed quant.
- KV cache GPU dinamica (cap iniziale 1024, crescita ×2, recapture graph).
- Partial offload `--gpu-layers N` funzionante in `chat`/`generate`/`serve`
  (ibrido GPU-prefix/CPU-suffix, corretto ma ~0.3 tok/s sul 1.5B — v1 lento).
- Batched prefill ON di default con test di parità; `SERIUS_PREFILL_BATCH=0`
  per forzare il path per-token.
- Server: listener con rebind su errore, ops port separata, `/v1/models`
  CRUD + `GET /v1/models/{id}` già implementato (server.cpp:675).
- Moduli sperimentali spostati in `experimental/` (opt-in CMake).
- GGUF loader: enum GGMLType completo (IQ/TQ/MXFP4/NVFP4/F8), check
  required-tensors, `qmap` QuantWeight per decode packed.

## Gap misurati vs llama.cpp (stesso HW: RTX 3060 Laptop 6GiB + i9-11900H)

| Area | Serius | llama.cpp | Rapporto |
|------|--------|-----------|----------|
| Decode GPU | vedi BENCHMARKS.md | — | ~1.0x (parità) |
| Prefill GPU | ~5-8k tok/s | ~19-24k tok/s | 0.27-0.37x |
| Decode CPU | 17.8 tok/s (SmolLM2 Q4_0) | 67.4 tok/s | 0.26x |

## Todo — in ordine di impatto

### 1. CPU decode: riverificare + ottimizzare GEMV Q4_K [PRIORITÀ]
- **Fatto (2026-11-23)**: `q4k_row_avx512` implementato in
  `src/blas/matvec_quant.cpp` e instradato in `pick_row_fn` per
  `QuantType::Q4_K`. Verificato con test standalone (err ~1e-8, ~2-5x
  vs AVX2) e build ninja verde.
- **Misura CPU post-kernel**: Qwen2.5-0.5B Q4_K_M `--gpu cpu` =
  **6.8 tok/s** (48 token) — nel range storico 1.9-8. Il modello è
  quasi tutto Q4_1/Q8_0 (in=896 non è multiplo di 256 → Q4_K convertito
  in Q4_1 dal loader), quindi il kernel Q4_K ha impatto limitato qui.
  **Il collo di bottiglia CPU reale è altrove**: candidati = GEMV
  Q4_1/Q8_0 (già AVX-512 ma forse non row2-paired), lm_head
  (vocab 151936×896 ≈ 136M parametri — ~25% del modello, gira a ogni
  token), attention CPU, o mancanza di thread-pool persistente.
  Profilare con `SERIUS_PROF=1` (breakdown `[prof]` ogni 300 token:
  qkv/o/gateup/down/lm) per identificare il componente dominante.
- **Profiling `SERIUS_PROF=1` (Qwen 0.5B Q4_K_M, 310 token)**: MLP
  (gate_up+down) = **66%** del forward (~57ms/token), lm_head ~17.5ms
  (21%), attention ~12ms (13%). Bandwidth efficace ~2.8GB/s vs ~30GB/s
  disponibili → kernel latency-bound, non memory-bound.
- **Test effettuati**: `SERIUS_INTDOT=1` → 5.5 tok/s (peggio di 6.8
  baseline, int8-dot non aiuta qui); `OMP_NUM_THREADS=1` → 1.0 tok/s
  (OpenMP funziona, scala ~7-10x). Q4_K AVX-512 ha impatto limitato su
  questo modello (in=896 non multiplo di 256 → Q4_K→Q4_1 nel loader).
- **Tentativo row2-pairing (2026-11-23, RISULTATO NEGATIVO)**:
  implementati `q41_row2_avx512`/`q80_row2_avx512` (schema di
  `q40_row2_avx512`: 8 accumulator per riga, x condiviso, 2 stream di
  pesi). Correttezza verificata standalone (~1e-8 con metrica
  sum_abs), ma micro-bench A/B (`bench_row2` vs `bench_norow2`,
  oggetti separati) ha mostrato **Q8_0 ~60-100% più lento** a
  in=2048 e Q4_1 neutro/peggiore — probabilmente register pressure
  (16 zmm acc + dequant temps → spill). **Rimosso**; dispatch
  ripristinato a Q4_0-only. Non riprovare senza profilo
  (VTune/perf) o con kernel a 16 acc/riga.
- **Contesa attiva**: durante queste misure girava `serius stress -t
  soak -d 14400` (soak 4h, 4 user) + `serius serve` su porta 8080 —
  i benchmark sono rumorosi; le conclusioni negative sopra sono su
  segnali grandi (>50%), i neutri vanno riverificati a sistema scarico.
- **Q4_K int-dot implementato (2026-11-23, opt-in)**:
  `q4ki_row_avx512` in `matvec_quant.cpp` — maddubs su 64 weight/coppia
  di sub-blocchi, scale/min 6-bit via `q4k_scale_min`, correzione
  `-dmin*m6*xq.c`. Dispatch `SERIUS_INTDOT=1` esteso a Q4_K (richiede
  in%256==0, solo AVX-512; altrimenti ricade su fp_path). Correttezza
  verificata vs riferimento fp64 (err ~1e-3 = errore attivazione int8,
  atteso). **Micro-bench: ~25-35% PIÙ LENTO del kernel fp32 AVX-512**
  (0.88 vs 1.17 ms, in=4864) — le riduzioni orizzontali per sub-blocco
  e la mancanza di VNNI su Rocket Lake lo penalizzano. Lasciato opt-in:
  su CPU con AVX512-VNNI (Sapphire Rapids+) potrebbe vincere —
  ricentare lì. Per Q8_0 int-dot NON implementato: q8_0 ha scale per
  32-weight e nessun min, ma serve variante saturation-safe
  (maddubs u8×i8 con q+128 overflow i16) — solo se profiling lo conferma.
- **Prossimo target CPU realistico**: le misure saturano ~6.3GB/s;
  il kernel fp32 Q4_K AVX-512 è già buono. Il gap vs llama.cpp
  (~4x) è probabilmente nel LOOP: troppi kernel launch/OpenMP sync
  per layer (24 layer × ~6 GEMV = ~150 lanci/token), niente fusion
  dei bias+norm, e lm_head pieno (151936×896). Considerare:
  (a) fusion gate+up già presente? verificare; (b) lm_head top-k o
  chunked; (c) thread-pool persistente se i lanci OpenMP contano.
- **Ipotesi gap CPU**: kernel fp32-dequant ~0.44 instr/peso vs
  llama.cpp che usa maddubs int-dot (~4x più efficiente); servirebbe
  int-dot path per Q4_1/Q8_0 (quello attuale copre solo Q4_0/Q4_1 con
  attivazione Q8_1 e Q4_K non è coperto per niente) oppure kernel
  con dequant in int16 e moltiplicazione intera.
- **Da fare**: estendere `SERIUS_INTDOT` a Q8_0 e misurare con
  `q8_1` activation su Q4_1 (già supportato — verificare perché non
  default); valutare kernel Q4_K int-dot; se lm_head >15% → logits
  solo su candidati top-k o lm_head nativo packed. Ultima misura
  era 0.23 tok/s (3.8s/token) MA con sessione parallela che lanciava test in
  loop — contesa CPU probabile, non necessariamente regressione.
- Comando: `SERIUS_PROF=1 ./build-ninja/src/serius.exe generate --model
  models/smollm2-q4/SmolLM2-135M-Instruct-Q4_0.gguf --gpu cpu --prompt Hi
  --max-tokens 32 --temperature 0`. Il profiler stampa breakdown ogni 300
  forward (`[prof]` su stderr).
- Se confermato ~17 tok/s: gap reale = kernel. `src/blas/matvec_quant.cpp`
  `pick_row_fn()`: Q4_0/Q4_1/Q8_0 hanno AVX-512; **Q4_K/Q5_K solo AVX2 o
  scalar** — Q4_K_M è il quant più diffuso, è il target.
- Implementare `q4k_row_avx512` (layout: superblock 256w/144B, 8 sub-blocchi
  32w, scale/min 6-bit packed — vedi `q4k_scale_min` e `q4k_row_scalar`).
- Anche valutare `SERIUS_INTDOT=1` (int8-dot path, attualmente opt-in,
  misurato parity su Q4_0/Q4_1 — riverificare su Q4_K dopo kernel AVX-512).
- Verifica: stesso comando generate, confrontare tok/s; output coerente;
  test di parità se esistenti per matvec_q.

### 2. Continuous batching nel server HTTP
- Oggi una request alla volta per modello (`src/server/server*.cpp`).
- Serve: sessioni KV multiple per modello (KVCache già istanziato per
  LoadedModel — serve un pool/slot per request), scheduler decode
  round-robin o chunked, SSE per-request isolato.
- Rischio: CUDA graph è legato a (model, cache) — cache multipli =
  graph ctx per slot o path eager per slot extra.
- Partenza: leggere `handle_chat_completions` in server.cpp e
  `KVCache`/`llama_forward_single_token` dispatch.

### 3. Prefill GPU batched (gap 0.3x)
- Batched prefill esiste (`SERIUS_PREFILL_BATCH` ON) ma usa pesi densi
  FP16/cublas path; llama.cpp usa GEMM quantizzato tiling.
- Opzioni: dequant scratch FP16 riusabile (menzionato in BENCHMARKS.md),
  o kernel GEMM packed. Misurare con `serius bench --latency-only`.

### 4. Partial offload performance (attuale ~0.3 tok/s, più lento di CPU)
- `src/infer/llama_gpu_offload.cpp` + suffix CPU in `llama_inference.cpp`.
- Fix: scratch persistente (oggi decine di cudaMalloc/layer/token),
  attention FP16, pesi quantizzati packed nel prefisso GPU.
- Solo dopo item 1: se CPU arriva a ~parità llama.cpp, il partial diventa
  meno urgente.

### 5. Packaging + release
- Installer Windows (es. WiX/NSIS o zip + script), bundlingCUDA runtime
  o requisito documentato, `docs/installation.md` aggiornato, versione
  in CMake (`--version` oggi 0.0), changelog.
- Igiene: `.gitignore` per *.obj/*.exe/*.log; verificare che `experimental/`
  non finisca nei package.

### 6. Stabilità
- `ctest -j8` flaky: timeout o serializzazione test pesanti (individuare
  quali — probabilmente test CUDA/inferenza).
- Soak test server: `src/stress/stress_runner.cpp` esiste — eseguire e
  fixare leak/crash.
- Rebind listener già fixato (server.cpp run()); verificare con churn.

### 7. Parità API OpenAI residua
- `GET /v1/models/{id}` esiste — fare test HTTP live e marcare todo.
- Verificare: errori standard (`error.message/type/code`), `usage`
  fields, `logprobs`/`top_logprobs`, `n>1`, `stop` sequences multiple,
  `presence/frequency_penalty`, `response_format` base.
- WebSocket non richiesto per ora.

### 8. Unificazione path inferenza
- `src/infer/` (produzione GGUF) vs `src/inference/` (autograd didattico).
- Decisione: marcare `src/inference/` come deprecated/experimental o
  migrare consumer. Non urgente per il prodotto ma riduce superficie.

## Avvertenze operative

- **Windows exe lock**: `serius.exe` in esecuzione blocca il link
  (LNK1104). Prima di rebuild: `taskkill //F //IM serius.exe` o controllare
  `tasklist //FI "IMAGENAME eq serius.exe"`. Sessioni parallele di agenti
  possono spawnare test — riprovare il link se fallisce.
- **Contesa misure**: benchmark CPU/GPU vanno fatti a sistema scarico;
  altre sessioni agenti invalidano i numeri.
- **CUDA graph**: i puntatori K/V sono catturati nel graph — mai
  reallocare KV senza `graph_ctx_destroy`/ricapture. `KVCache::instance_id`
  è la guardia anti-stale.
- **Non committare**: working tree contiene modifiche miste di più agenti;
  il proprietario decide i commit.
- Build: `cmake --build build-ninja` (Ninja+MSVC, `SERIUS_USE_CUDA=ON`).
  Le sorgenti `experimental/` richiedono `-DSERIUS_EXPERIMENTAL_*=ON`.
- Test: `ctest --test-dir build-ninja --output-on-failure`.

## Metriche di successo per "vendibile"

- CPU decode ≥ 0.7x llama.cpp stesso HW (oggi 0.26x).
- Prefill ≥ 0.6x llama.cpp.
- Server regge ≥4 richieste concorrenti con batching (oggi 1).
- `serius serve` + installer funzionante su Windows pulito.
- Zero crash noti su soak 30min; test suite verde `ctest -j8`.
