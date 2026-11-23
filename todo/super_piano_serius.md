# Super Piano Serius — percorso a prodotto reale

Data: 2026-11-23. Sintesi di `AUDIT_STUBS_AND_GAPS_2026-11-22.md`,
`BENCHMARKS.md`, `todo/roadmap_prodotto_2026-11-23.md` e dei finding
recenti. Documento di governo: ogni workstream ha obiettivo, task
concrete (file reali), criterio di done e dipendenze.

## 1. Dove siamo (numeri misurati, RTX 3060 Laptop 6GiB + i9-11900H)

| Area | Serius | llama.cpp | Rapporto | Giudizio |
|------|--------|-----------|----------|----------|
| Decode GPU (graph) | 340-470 tok/s | 294-369 tok/s | ~1.0-1.27x | **Vantaggio** |
| Decode GPU (eager) | ~158 tok/s | ~294-362 tok/s | ~0.5x | Gap |
| Prefill GPU | ~2.1-5k tok/s | ~9-19k tok/s | 0.27-0.37x | Gap strutturale |
| Decode CPU | 6.8-8.5 tok/s | ~25-67 tok/s | 0.26-0.35x | Gap strutturale |
| Partial offload | ~0.3 tok/s | — | rotto | **Bloccante** |
| Server | 1 req/volta | batching | — | Gap |

Punti di forza già acquisiti: CUDA graph decode (2-3x), batched prefill
default ON con cache dequant FP16, GGUF loader quant coverage estesa,
moduli "giocattolo" resi reali o onestamente marcati `experimental/`,
benchmark onesti (`serius bench`), test suite ~712 verde.

## 2. Principio guida

> Nessuna feature esposta deve essere peggiore dell'alternativa onesta.
> Se non possiamo renderla reale, la maschiamo o la etichettiamo.
> Ogni numero pubblicato deve essere misurato, non stimato.

## 3. Workstream ordinati

### WS-A — Partial offload: sistemare o seppellire (PRIORITÀ MASSIMA)

**Perché bloccante**: `--gpu-layers N` è esposto in `chat`, `generate`,
`serve`. A ~0.3 tok/s è peggio del CPU pieno — l'utente che lo prova
conclude che Serius non funziona.

**Target**: decode ibrido ≥ CPU-only sullo stesso modello (layer GPU
dovrebbero *aiutare*, non rompere).

**Analisi tecnica** (`src/infer/llama_inference.cpp:1284-1310`,
`src/infer/llama_inference_cuda.cpp` `llama_forward_single_token_cuda_impl`,
`src/infer/llama_gpu_offload.cpp`):

- Oggi il prefisso GPU gira in **FP32 denso** con ~10+ allocazioni
  device per layer per token (i buffer `normed_buf`/`query_buf`/… sono
  `Tensor({1,N})` + `alloc_gpu()` dentro il loop → cudaMalloc ripetuti).
- L'attenzione prefisso usa il branch non-fused (score + softmax +
  batched_matmul per ogni token) — niente fused-GQA.
- I pesi del prefisso sono **FP32/densi**, non i packed quant usati dal
  decode full-GPU: il prefisso paga il bandwidth pieno.
- Il D2H della hidden `{1,hidden}` per token è piccolo — non è il
  collo; sono i kernel.

**Task**:

1. **Scratch persistente** in `llama_forward_single_token_cuda_impl`:
   spostare le allocazioni `{1,N}` fuori dal layer loop (oBetter: un
   `DecodeScratch` per (model, layer_range) in cache, come
   `quant_prefill_scratch`). Ogni cudaMalloc ~10-20µs × ~10 buffer ×
   N layer × token = millisecondi sprecati.
2. **Prefill batched per il prefisso**: `llama_forward_inference` con
   partial layers oggi fa loop per-token su TUTTI i layer
   (`llama_inference.cpp:1140` esclude esplicitamente il batch).
   Serve un path ibrido: prefisso GPU batched (riusa
   `llama_forward_cuda_impl` con layer_begin/end) → suffix CPU batched.
   TTFT attuale è il vero killer per prompt lunghi.
3. **Pesi packed nel prefisso**: se i layer GPU usano lo stesso
   `use_quant`/`qproj` del decode full-GPU, il prefisso viaggia a
   bandwidth quantizzato. Richiede che `offload_llama_model_partial`
   carichi anche `qkv_w`/`gate_up_w`/`o_w`/`down_w` packed (oggi tiene
   solo FP32 dual-resident).
4. **Graph parziale**: il CUDA graph può catturare il solo prefisso
   (input token+pos, output hidden D2H) — il suffix CPU resta eager.
   Kernel catturati = ~70% del layer loop → gran parte del beneficio
   graph senza duplicare KV logic.
5. **Misura**: `serius generate --gpu-layers 8` su Qwen2.5-Coder-1.5B
   (non entra in 6GiB) → target ≥2 tok/s (vs 0.3 oggi, vs ~1.3 CPU).

**Alternativa se WS-A fallisce**: rimuovere `--gpu-layers` da CLI/serve
e documentare "partial offload non supportato" — meglio di una flag
velenosa. Decisione dopo il task 1 (se scratch da solo porta a >1 tok/s
il fix completo è realistico).

### WS-B — Decode GPU eager: copertura graph (PRIORITÀ ALTA)

**Perché**: il graph è il nostro vantaggio competitivo (2-3x). Oggi è
silenziosamente non eleggibile per intere classi di modelli.

**Task**:

1. **Log esplicito** in `llama_forward_single_token_cuda_graph` quando
   l'elegibility fallisce (una volta per modello+cache, non per token):
   "CUDA graph non disponibile: <motivo>" — dtype, missing packed
   weights, lm_head. Oggi il fallback è invisibile.
2. **Estendere elegibility**: audit dei `dev_kind`/`type` supportati dai
   kernel decode (`gemv_quant_*`) vs quelli rifiutati dal check attuale
   (`llama_inference_cuda.cpp:~870-881`). Se Q4_0/Q8_0/Q5_K hanno kernel
   decode validi, allargarli.
3. **Graph per modelli non-quantizzati**: per FP16 denso il decode usa
   `gemv_kn_sk_f16` — kernels catturabili. Il check `dtype==Float16 &&
   packed` esclude tutto il path denso; valutare una variante graph
   per il branch `proj()` non-quant (serve solo che i weight siano
   GPU-resident).
4**Misura**: `SERIUS_NO_GRAPH=1` vs default su ogni modello di test;
   obiettivo eager ≥0.8x llama.cpp e graph sempre attivo quando possibile.

### WS-C — CPU decode: gap strutturale (PRIORITÀ, già in corso dall'utente)

**Stato**: 6.8-8.5 tok/s vs llama.cpp ~25-67 (0.26-0.35x). Profiling:
MLP 61%, lm_head 21%, attention 13%; ~6.3 GB/s effective vs ~30 teorici.
Kernel-level tweaks (int-dot, row2, prefetch) tutti negativi — il gap è
**strutturale**: servirebbe un loop stile ggml (thread persistenti,
fusion, scheduling).

**Task** (in ordine di impatto/realismo):

1. **lm_head (~21%)**: per greedy/top-k si possono calcolare logits
   solo sui candidati o usare un lm_head chunked con early-exit —
   oppure tenere lm_head in FP16 packed su CPU. ~5.3ms/tok → target <3.
2. **Fusion elementwise MLP**: `mlp rest` 6.6ms/tok per silu/bias/add —
   fondere in kernel dedicato (gate_up già fused; aggiungere silu+residual).
3. **Thread-pool persistente**: contare le barrier OpenMP per token
   (layer × ~6 GEMV + elementwise × 2): se >100 sync/token, un pool
   con split fisso risparmia launch+sync.
4. **Rivalutare architettura**: se 1-3 portano a ~12-15 tok/s, il gap
   residuo è ~2x — accettabile documentato; inseguire llama.cpp 1:1 su
   CPU è un progetto ggml-style, non un fix.
5. **Non ripetere**: row2-pairing (documentato negativo), prefetch
   software (negativo), int-dot senza VNNI (negativo su Rocket Lake).

### WS-D — Prefill GPU (gap 0.27x)

**Stato**: `prefill_dq_cache` (cache FP16 dequant) ha già dimezzato il
cold→warm (0.61s→0.26-0.39s su ~600 tok). llama.cpp usa quantized GEMM
tiling diretto.

**Task**:

1. **Misura pulita warm**: `serius bench` con prompt ~512-1024 tok,
   ≥3 run, sistema scarico → baseline reale post-cache.
2. **Cache completa all'offload**: per modelli che stanno in VRAM,
   pre-dequantizzare TUTTI i pesi in FP16 al load (memoria ~2x quant
   ma prefill = GEMM denso puro, niente dequant per-layer). Su 6GiB
   va bene per ≤1.5B Q4_K_M; opzione `--prefill-dense` o auto se fits.
3. **GEMM packed diretto**: solo se 1-2 non bastano — kernel tipo
   llama.cpp `mul_mat` quantizzato. Progetto grosso, ultimo target.
4.**Misura**: target ≥0.6x llama.cpp pp512 (oggi 0.27-0.37x).

### WS-E — Server: continuous batching (PRIORITÀ MEDIA)

**Stato**: 1 request alla volta per modello. KVCache già per-sessione
in `LoadedModel` — manca il pool di slot e lo scheduler.

**Task**:

1. **Slot pool**: N `KVCache` per `LoadedModel` (cap di default 4-8,
   configurable). La graph ctx è keyed (model, cache instance) →
   servirà graph ctx per slot o eager per slot extra (vedi WS-B).
2. **Scheduler decode**: round-robin 1 step/token/request oppure
   chunked-prefill (prompt lunghi yield). Iniziare round-robin —
   semplice e corretto.
3. **Prefill**: può restare serializzato (è GPU-bound burst); il
   batching decode è dove conta la latenza percepita.
4. **SSE**: verificare che `handle_chat_completions` stream isolato per
   request non blocchi le altre (thread per request esistente?).
5. **Misura**: 4 req concorrenti → target ~3-4x throughput vs seriale.

### WS-F — Packaging + stabilità (PRIORITÀ MEDIA)

- **Fatto parziale**: `scripts/package.ps1` ( MSI + zip + sha256) esiste
  e tiene aggiornato; `SERIUS_PATH` fixato.
- **Da fare**: testare il MSI/zip su Windows pulito (VM o utente
  separato), bundling CUDA runtime o requisito documentato,
  `docs/installation.md` allineato, versione `0.1.0` → `0.2.0` quando
  WS-A chiuso.
- **Stabilità**: `ctest -j8` flaky noti (E2E timeout, file-lock
  OneDrive) — serializzare i test pesanti o alzare timeout;
  soak 30min server con `stress_runner` dopo WS-E.
- **Igiene**: `.gitignore` per `*.obj/*.exe/*.log/dist/`; `nul` file
  da rimuovere.

### WS-G — Parità API OpenAI residua (PRIORITÀ BASSA)

- `GET /v1/models/{id}` esiste — serve solo test HTTP live.
- Verificare: `error.message/type/code`, `usage`, `logprobs`/`top_logprobs`,
  `n>1`, `stop` multipli, `presence/frequency_penalty`, `response_format`.
- Decisione su `src/inference/` didattico: marcare deprecated o
  migrare — riduce superficie, non bloccante.

## 4. Sequenza consigliata

```
Fase 1 (bloccante):   WS-A partial offload → fix o rimozione onesta
Fase 2 (vantaggio):   WS-B graph coverage + log elegibility
Fase 3 (concurrente): WS-C CPU (gemelli kernel: utente) — non toccare
                      matvec_quant.cpp mentre ci lavora
Fase 4:               WS-D prefill misure + cache densa
Fase 5:               WS-E continuous batching
Fase 6:               WS-F packaging + soak + WS-G API
```

Regola di collisione: **WS-C è territorio utente** (kernel CPU attivi);
gli agenti lavorano su WS-A/B/D/E che toccano `llama_inference_cuda.cpp`,
`llama_gpu_offload.cpp`, `server.cpp` — coordinare via commit frequenti.

## 5. Rischi

- **VRAM 6GiB**: cache FP16 densa + graph ctx per slot può esplodere —
  sempre fallback su OOM (pattern già usato in `prefill_dq_cache`).
- **Graph ctx per slot** (WS-E): cattura per ogni KVCache = overhead
  fisso; valutare graph solo per slot 0 e eager per gli altri.
- **Misure rumorose**: soak/serve attivi invalidano i benchmark —
  fissare regola "misure solo a sistema scarico" o lanciare
  `stress -t soak` solo di notte.
- **OneDrive lock**: file in `OneDrive\Desktop` vanno in lock durante
  sync — rebuild con retry, mai `rm` su file aperti.

## 6. Comando di verifica rapido (tutto il piano)

```bash
# Build
cmake --build build-ninja

# Suite
ctest --test-dir build-ninja --output-on-failure -j4

# Benchmark attesi (sistema scarico)
./build-ninja/src/serius.exe bench -m models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf --latency-only -n 128
# decode atteso ≥340 tok/s (graph), ~158 (SERIUS_NO_GRAPH=1)

# Partial offload (WS-A done = questo deve essere >1 tok/s)
./build-ninja/src/serius.exe generate --gpu-layers 8 -m models/Qwen2.5-Coder-1.5B-Instruct... 
```
