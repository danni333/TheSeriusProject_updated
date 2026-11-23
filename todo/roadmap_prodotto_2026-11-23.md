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
  (ibrido GPU-prefix/CPU-suffix, corretto ma ~0.3 tok/s sul 1.5B — **rotto
  di fatto, priorità alta: sistemare o mascherare** — vedi item 4).
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
  ricentare lì.
- **Q8_0 int-dot implementato (2026-11-24, opt-in)**:
  `q80i_row_avx512`/`q80i_row_avx2` in `matvec_quant.cpp`. I pesi Q8_0
  sono già int8 con segno → niente nibble extract; si usa
  `madd_epi16` su una copia i16 dell'attivazione (`XQuant::q16`,
  generata in `quantize_x_q8_avx2`) — intero esatto, **nessuna
  saturazione** (la variante maddubs con offset +128 potrebbe
  overfloware i16, scartata). Dispatch `SERIUS_INTDOT=1` ora copre
  Q4_0/Q4_1/Q8_0/Q4_K (K%32==0). Corretto vs riferimento scalare
  (err = quantizzazione attivazione int8; AVX-512 e AVX2 identici).
  **Misura NON conclusiva**: soak 4h attivo durante il lavoro → numeri
  inutilizzabili; riverificare a sistema scarico. Nota: con
  `SERIUS_INTDOT=1` `MatvecQuantK.Q4KMatchesReference` fallisce per
  tolleranza (test tarato sul path fp) — tradeoff di precisione noto,
  non un bug.
- **AVX512-VNNI: la CPU CE L'HA (2026-11-24)** — CPUID leaf7.0:ECX
  bit11=1 su i9-11900H (la nota "mancanza di VNNI su Rocket Lake" era
  sbagliata: manca solo AVX-VNNI VEX, non AVX512-VNNI EVEX).
  `CpuInfo::has_avx512_vnni` aggiunto (dispatch.cpp cpuid). Kernel
  `q40vnni_row_avx512`/`q41vnni_row_avx512`/`q80vnni_row_avx512`:
  `vpdpbusd` fonde maddubs+madd in 1 istruzione; Q8_0 usa offset +128
  sui byte (sicuro: accumulo i32, niente saturazione i16). Con
  `SERIUS_INTDOT=1` il dispatch preferisce VNNI se supportato.
  Output VNNI identici al path non-VNNI (stessa attivazione int8).
  Q4_K VNNI non implementato.
- **Misura VNNI a sistema scarico (2026-11-24, i9-11900H, micro-bench
  `%TEMP%\mvq_bench`)**: VNNI **PERDE** ovunque — q41 4.3 vs 6.8 GB/s,
  q80 7.4 vs 14.8, q40 15.4 vs 15.3 (intdot non-VNNI). Dispatch corretto:
  kernel VNNI solo con `SERIUS_VNNI=1` esplicito. **Scoperta
  collaterale**: int-dot Q8_0 (madd_epi16) è **2.2x più veloce del path
  fp** (14.8 vs 6.7 GB/s) — unico tipo dove int-dot vince; Q4_0/Q4_1
  restano più veloci in fp (18.9/11.5 vs ~15/7 GB/s). Candidato:
  abilitare int-dot Q8_0 di default (stessa classe di precisione del
  Q8_1 di llama.cpp) dopo verifica end-to-end tok/s.
- **Misure pulite a sistema scarico (2026-11-23 notte, prima del
  throttle termico)**: Qwen2.5-0.5B Q4_K_M `--gpu cpu` = **8.49 tok/s**
  AVX-512 default; `SERIUS_NO_AVX512=1` (AVX2) = 7.21; `SERIUS_INTDOT=1`
  = 8.25 (parità, non abilita). **Attenzione**: il laptop throttla in
  ~10 min di carico sostenuto — run consecutivi danno 5-6 tok/s;
  confrontare solo run ravvicinati o brief.
- **Profiling pulito (310 token)**: MLP=50.3ms/tok (gu=33.5, down=15.6),
  head=16.5ms/tok, attention=11.2ms/tok; fwd=82.5ms/tok + sampler
  ~11ms → ~10 tok/s con thread caldi.
- **Tentativo software-prefetch (2026-11-23, RISULTATO NEGATIVO)**:
  `prefetch_quant_weight` (dichiarato in blas.h ma MAI chiamato) ora
  invocato per i pesi del layer successivo nel decode loop
  (`llama_inference.cpp`, opt-in `SERIUS_PREFETCH=1`, default OFF).
  Misura: neutro/peggiore su i9-11900H — le ~190K prefetch/layer
  costano quanto il beneficio. Tenere il codice per ri-test su altri
  uarch; NON abilitare di default senza misura pulita.
- **Prossimo target CPU realistico**: ~6.3GB/s effective vs theoretical
  ~30GB/s; kernel fp32 Q4_K AVX-512 già buono. Il gap vs llama.cpp
  (~4-7x) è strutturale: serve un loop stile ggml (thread persistenti,
  kernel AVX2 ottimizzati alla loro maniera) — progetto a sé, non
  kernel-level tweak. lm_head (21%) e attention (13%) sono i secondi
  target dopo MLP (61%).
- **Ipotesi gap CPU**: kernel fp32-dequant ~0.44 instr/peso vs
  llama.cpp che usa maddubs int-dot (~4x più efficiente); servirebbe
  int-dot path per Q4_1/Q8_0 (quello attuale copre solo Q4_0/Q4_1 con
  attivazione Q8_1 e Q4_K non è coperto per niente) oppure kernel
  con dequant in int16 e moltiplicazione intera.
- **Da fare**: ~~estendere `SERIUS_INTDOT` a Q8_0~~ ✅ fatto 2026-11-24
  (mancano solo la misura a sistema scarico e l'eventuale VNNI);
  `q8_1` activation su Q4_1 già supportato — non default perché misurato
  parity (8.25 vs 8.49 tok/s); se lm_head >15% → logits
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
- **Fatto (2026-11-23 notte)**: `quant_prefill_gemm` dequantizzava +
  traspondeva OGNI peso packed a OGNI prefill. Aggiunta cache FP16
  dequantizzata persistente (`prefill_dq_cache` in
  `llama_inference_cuda.cpp`, fingerprint dims/type/byte-samples,
  lazy alloc, fallback su OOM). Misura serve ~600 tok prompt + 8 out:
  **0.61s cold → 0.26-0.39s warm** (~2x). Beneficio solo con ≥2
  prefill per processo (server/chat multi-turn/bench) — `generate`
  one-shot non ne trae vantaggio. Prossimo: estendere a terza+ run
  e misurare tok/s prefill pura; valutare se conviene cache FP16
  completa all'offload per modelli che ci stanno in VRAM.
- Opzioni residue: GEMM packed diretto (kernel tiling tipo ggml),
  o dequant peak-memory più basso. Misurare con `serius bench
  --latency-only`.

### 4. Partial offload performance — ⛔ DA FARE (feature shipped rotta)
- `src/infer/llama_gpu_offload.cpp` + suffix CPU in `llama_inference.cpp`.
- **Stato**: ~0.3 tok/s sul 1.5B — *più lento del CPU pieno* (~1.3 tok/s).
  Una feature esposta (`--gpu-layers` in chat/generate/serve) che in
  pratica non funziona: o si sistema o si maschera onestamente.
- Cause identificate: decine di cudaMalloc/layer/token nel prefisso GPU
  (path FP32 non ottimizzato), attention FP32 eager, prefill sempre
  per-token (niente batched), nessuna quantizzazione packed nel prefisso.
- Fix previsti: scratch persistente (KV + buffers riusati), attention
  FP16/quantized nel prefisso, pesi packed quant anche nel prefisso GPU,
  prefill batched per i layer GPU. Target: ≥ CPU-only a parità di layer.
- Alternativa onesta se il fix è troppo costoso: rimuovere il flag o
  marcarlo experimental con warning esplicito al load.
- Da NON fare: sedersi sui fix parziali — il path ibrido ha costi di
  sincronizzazione D2H per token che vanificano benefici piccoli.

### 4b. Decode GPU eager (Senza graph): copertura e gap — DA FARE
- Misurato 2026-11-23: `SERIUS_NO_GRAPH=1` sul Qwen2.5-0.5B Q4_K_M =
  **~158 tok/s** vs ~340-470 con graph e ~294-362 llama.cpp eager.
  Il CUDA graph vale ~2-3x: e' il vero vantaggio competitivo attuale.
- Il graph è eleggibile solo se **tutti** i layer hanno pesi packed FP16
  su GPU + lm_head_q. Modelli FP16 densi, Q4_0/Q8_0 su certi path, e
  **partial offload** ricadono sempre in eager.
- Lavoro: (a) estendere l'elegibility ai quant type mancanti se i kernel
  decode li supportano già; (b) valutare graph per il prefisso GPU del
  partial offload (le layer CPU non entrano nel graph ma il prefisso sì);
  (c) se un modello non è eleggibile, log esplicito del motivo (oggi
  silenzioso → l'utente crede di usare il graph e non lo usa).

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

## Profiling CPU decode — 2026-11-23

Setup: `SERIUS_NO_GPU=1 SERIUS_PROF=1 serius bench -m
models/smollm2-q4/SmolLM2-135M-Instruct-Q4_0.gguf -n 320 --latency-only`
(RTX 3060 laptop presente ma disabilitata; CPU = dotato AVX-512 del
laptop). Decode misurato: **25.5 tok/s** (30 tok: 29.0 tok/s — la
differenza e' warm-up/context growth).

Breakdown `SERIUS_PROF` (somma su 300 token, ms/token = /300):

| Componente | ms/tok | % di ~39ms/tok |
|---|---:|---:|
| MLP totale | 18.8 | ~48% |
|   - gate_up GEMV | 11.7 | ~30% |
|   - down GEMV | 6.4 | ~16% |
|   - mlp rest (silu/bias/add) | 6.6 | ~17%* |
| Attention totale | 10.5 | ~27% |
|   - pre (norm/qkv proj) | 5.0 | |
|   - head (attn+softmax+o) | 2.7 | |
|   - post | 2.8 | |
| lm_head GEMV | 5.3 | ~14% |

GEMV totali (qkv+o+gateup+down+lm): **~30ms/tok ≈ 77% del decode**.

Conseguenze:

1. Decode e' **bandwidth-bound sui kernel GEMV quantizzati** — coerente
   col ~6.3 GB/s misurato. Il gap vs llama.cpp non e' in un singolo
   kernel ma nella somma: piu' layer × piu' launch OMP × sync.
2. **lm_head e' ~14%** (vocab 49152 × hidden 576 = 28M pesi, ~21% dei
   parametri del modello). Candidato: per top-k/top-p/greedy si possono
   calcolare solo i logits necessari (chunked top-k o early-exit),
   oppure tenere lm_head in fp16 su CPU.
3. **mlp rest 6.6ms/tok** e' alto per elementwise — verificare se
   silu/bias/residual non sono fusi (gate_up e' gia' fused).
4. Attention ~27%: con seq corta (10-50 tok) e' overhead di
   kernel/launch, non FLOPs — candidati a fusione o kernel dedicato
   decode a 1 token.
5. **Prefill non e' il collo di bottiglia a questo scale**: prompt 47
   tok → TTFT 1325ms = **35.5 tok/s prefill > 28.8 decode** (batch GEMM
   batched e' piu' efficiente del GEMV decode). Il 13 tok/s della run
   precedente era warm-up/noise. Da riverificare a prompt molto piu'
   lunghi (attention O(n^2) inizia a contare oltre ~1k tok).

Prossimi step consigliati (in ordine d'impatto):
- ~~Misurare OMP threads~~ **FATTO**: sweep OMP_NUM_THREADS su i9-11900H
  (8C/16T): 6→23.3, **8→32.0**, 10→30.9, 16→~25 tok/s decode. I core
  fisici vincono — decode e' memory-bound e i sibling HT aggiungono solo
  overhead di barrier. Implementato `blas::physical_core_count()` +
  `configure_omp_threads()` (dispatch.cpp): default OMP = core fisici,
  `OMP_NUM_THREADS` esplicito vince. Chiamato in `main_cli`.
  Decode 24.8→~29 tok/s misurato (con contesa di fondo); ~32 a sistema
  scarico.
- AVX-512 vs AVX2 verificato: 24.8 vs 16.5 tok/s — AVX-512 resta
  preferibile anche su Rocket Lake (double-pumped ma win sui 512-bit
  accumulators).
- Sampler verificato gia' ok: top-k sui logits grezzi, solo k exp.
- Kernel GEMV Q4_0 AVX-512 gia' scritti bene (16 accumulator, prefetch,
  row2 pairing); restano candidati micro-ott: piega del termine -8*d*sx
  (correzione Q4_0) nel kernel — stimato ~10-15% kernel, rischio medio.
- Misurare prefill a prompt >1k tok (attention O(n^2)).

## Breakdown GEMV per tipo — misurato 2026-11-24 (SERIUS_PROF)

Setup: `SERIUS_NO_GPU=1 SERIUS_PROF=1 OMP_NUM_THREADS=8 serius generate
-m qwen2.5-0.5b-instruct-q4_k_m.gguf -n 300`. I valori `g_*` del report
sono somme su 300 token: divisi per 300 danno ms/tok. Decode ~8.1 tok/s,
fwd ~109.7 ms/tok.

| GEMV | tipo effettivo | ms/tok | % GEMV | byte/tok | GB/s |
|---|---|---:|---:|---:|---:|
| gate_up (fused) | Q4_1 | 48.3 | 46% | 130.7 MB | 2.71 |
| lm_head | Q4_0 | 24.2 | 23% | 76.6 MB | 3.17 |
| down | Q4_K(12 layer)+Q4_1(12) | 21.7 | 21% | ~62 MB | 2.86 |
| qkv (fused) | Q4_1 | 6.0 | 6% | 15.5 MB | 2.60 |
| o_proj | Q4_1 | 4.7 | 4% | 12.0 MB | 2.58 |
| **totale** | | **~105** | **96% di fwd** | ~297 MB | ~2.8 |

Nota: il GGUF "q4_k_m" e' un mix custom (prevalentemente Q5_0 + Q6_K/Q4_K
su down + Q8_0 su lm_head); il loader normalizza quasi tutto a Q4_1
(Q5_0/Q6_K non hanno kernel packed), Q4_K resta nativo dove in%256==0,
lm_head forzato a Q4_0. Quindi **Q4_1 e' il tipo che domina il decode**
(~74% dei byte), non Q4_K.

Conferme:
- time-share ~= byte-share → decode davvero bandwidth-bound.
- lm_head Q4_0 e' il kernel piu' efficiente/byte (3.17 GB/s): le 151936
  righe danno il miglior parallelismo OpenMP.
- GEMV = 96% del forward: norm/silu/residui/softmax insieme ~5 ms/tok.
- Priorita' kernel: gate_up (+10% = ~4.8 ms/tok) > lm_head (top-k
  early-exit o chunked) > down. Int-dot Q8_0 irrilevante su questo
  modello (nessun GEMV Q8_0 nel path dopo la fusion qkv).

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

- `--gpu-layers N` decode ≥ CPU-only sullo stesso modello (oggi ~0.25x —
  **bloccante**: una flag esposta che peggiora le cose non è vendibile).
- Decode GPU eager ≥ 0.8x llama.cpp (oggi ~0.5x senza graph).
- CPU decode ≥ 0.7x llama.cpp stesso HW (oggi ~0.35x a sistema scarico:
  8.5 vs ~25-30 stimati — riverificare).
- Prefill ≥ 0.6x llama.cpp.
- Server regge ≥4 richieste concorrenti con batching (oggi 1).
- `serius serve` + installer funzionante su Windows pulito.
- Zero crash noti su soak 30min; test suite verde `ctest -j8`.
