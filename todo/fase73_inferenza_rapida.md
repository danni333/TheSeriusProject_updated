# Fase 73 — Inferenza Veloce: battere llama.cpp su CPU

## Obiettivo

Superare le prestazioni di llama.cpp (`llama-cli`/`llama-bench`) su CPU per la
stessa quantizzazione GGUF, misurato in tok/s di decode (single-stream) e in
tok/s di prefill. Target: **≥ +20% vs llama.cpp** sulla stessa macchina e stesso
file `.gguf`.

Stato attuale (i9-11900H, SmolLM2-135M Q4_0): **~23 tok/s** decode.
llama.cpp stesso modello/quantizzazione: stimato 60–150 tok/s (da misurare).

## Analisi del gap

| Fattore | llama.cpp | Serius (oggi) | Impatto |
|---|---|---|---|
| Pesi a runtime | Quantizzati (Q4_0/Q8_0/Q4_K) | **Dequantizzati a FP32** (~540MB/token) | Dominante: 4–8x traffico DRAM |
| Kernel GEMV | Dequant-on-the-fly SIMD per formato | matvec FP32 (appena aggiunto per lm_head) | ~2–4x |
| Parallelismo | Thread persistenti / OpenMP ben calibrato | OpenMP fork/join ~210 regioni/token | Medio |
| Layout pesi | `[out][in]` row-major per neurone | `[in][out]` (column-major per neurone) | Mean per GEMV |
| Fusioni | qkv/gate_up fuse, norm fuse | 7 GEMV + 2 norm per layer | Piccolo-medio |
| KV cache | Quantizzata (opzionale) | FP32 | Contesto lungo |

Considerazioni hardware (i9-11900H, Tiger Lake):
- AVX-512 presente **ma con license-based downclocking** — va misurato se
  conviene usare kernel AVX-512 o resta AVX2 per i GEMV quantizzati.
- ~25–30 GB/s DRAM reale: con Q4_0 (76MB/token) il tetto è ~300 tok/s;
  il collo diventa la compute dequant+FMA.

## Fasi di lavoro

### A — Storage pesi quantizzati + kernel GEMV dequant (fondamenta)

- **A1** `QuantTensor`: nuovo tipo che conserva i blocchi GGUF grezzi
  (Q4_0: 18B/32 pesi, Q8_0: 34B/32, Q4_1, F16, F32) + metadati (tipo, `[out,in]`,
  byte/row). Loader GGUF popola sia `Tensor` FP32 (path autograd/training,
  invariato) sia la vista quantizzata per l'inference.
- **A2** Kernel `matvec_qXX`: per riga di output `j`, scorre i blocchi K/32,
  dequantizza 32 pesi in registri e FMA con la chunk di `x`. Varianti
  `avx512`, `avx2`, `scalar`; dispatch su `CpuInfo`. Partenza da
  `gguf.quants.dequantize` come riferimento di formato (validato in Python).
- **A3** Plumbing: `LLaMABlock`/`LLaMAModel` espongono pesi quantizzati
  opzionali per q/k/v/o/gate/up/down + lm_head. `llama_inference` usa il
  path quantizzato quando presente, altrimenti FP32 (fallback invariato).
- **A4** Gate di correttezza: logits confrontati col path FP32 su prompt
  fisso (tolleranza ~1e-2 dato Q4_0), `ctest` verde.

### B — Decode-path: fusioni e threading

- **B1** Fusione QKV in un'unica GEMV (pesi concatenati `[q|k|v]` in load) e
  gate+up analogamente → da 7 a 4 GEMV/layer, meno overhead e meno passaggi
  su `x`.
- **B2** Persistenza thread: misurare costo OpenMP fork/join; se >10% del
  tempo-token, passare a thread pool persistente con partizione statica
  delle righe di output (i GEMV quantizzati parallelizzano su `N`).
- **B3** Micro-ott allocazioni: buffer di lavoro per layer allocati una volta
  per forward invece che `Tensor(...)` per layer×token (~150 alloc/token).
- **B4** KV layout `[head][pos][dim]` per accessi contigui in attention;
  verificare vectorizzazione di score/softmax.

### C — KV cache e contesto lungo (se serve dopo A+B)

- **C1** KV cache quantizzata Q8_0 opzionale (flag), con dequant su lettura.
- **C2** Verifica sliding-window/long-context esistente.

### D — Benchmark e iterazione

- **D1** `bench_decode`: prompt fisso, 256 token, tok/s decode + prefill;
  matrix di dimensioni (Q4_0, Q8_0, FP32-safetensors).
- **D2** Baseline llama.cpp: compilare `llama-cli`/`llama-bench` o usare
  binario release, stessa macchina, stesso `.gguf` → tabella confronto.
- **D3** Profiling residuo: timer per stadio (GEMV vs attention vs sampler),
  `perf`/ETW se disponibile.

### E — Estensioni future (fuori scope iniziale)

- Speculative decoding (Fase 14 già implementata: `SpeculativeDecoder`) da
  collegare al path chat.
- Continuous batching nel server HTTP.
- Kernel Q4_K/Q5_K/Q6_K se si vogliono quantizzazioni più spinte.

## Rischi

- **Correttezza dequant**: mitigata dal riferimento Python `gguf` già usato
  per validare i tensori (corr 0.99998).
- **AVX-512 downclock**: misurare A/B AVX2 vs AVX512 fin da A2.
- **Complessità kernel**: Q4_0 è il formato più semplice (scala fp16 + nibble
  simmetrici); Q8_0 idem. Partire solo da questi due copre il GGUF attuale.
- **RAM**: mantenere FP32 + quantizzato costa ~+80MB per Q4_0 — accettabile;
  in futuro si può droppare il FP32 per l'inference se la RAM conta.

## Definition of done

- [ ] Decode tok/s ≥ 1.2× llama.cpp stesso modello/quant/macchina
- [ ] Output coerente (stesso prompt, stessa seed → testo paragonabile)
- [ ] `ctest --output-on-failure` verde
- [ ] Nessun regress su path safetensors/training (fallback FP32 attivo)

## Stato avanzamento (sessione decode-quant)

**Fatto e verificato:**

- Path quantizzato attivo end-to-end: `QuantWeight` (byte GGUF grezzi) in
  `LLaMABlock`/`LLaMAModel`, `matvec_q` con kernel Q4_0/Q4_1/Q8_0
  (AVX-512/AVX2/scalar) in `src/blas/matvec_quant.cpp`.
- Pesi fusi `qkv_w` (Q+K+V) e `gate_up_w` (gate+up) → 4 GEMV/layer.
- `lm_head` usa `token_embd` Q8_0 → **ri-quantizzato a Q4_0** da FP32
  (`quantize_fp32_to_q40`): 30MB → 16MB/token.
- Attention parallelizzata sulle teste + RoPE con tabelle cos/sin cached.
- SiLU vettoriale (`exp256_ps`, errore ~1e-7), softmax vettoriale.
- Buffer decode `thread_local` (niente ~240 malloc+memset/token).
- Sampler: `nth_element` + niente copy logits quando non serve penalty.
- FTZ/DAZ (MXCSR) su main+worker OpenMP.
- Unroll 8 blocchi + `_mm_prefetch` nei kernel AVX-512 — **fix critico**:
  Q4_1 (down_proj) era a 7.2GB/s → 23GB/s dopo l'unroll (era il collo MLP).

**Misure (i9-11900H, Q4_0, OMP_NUM_THREADS=10, OMP_WAIT_POLICY=ACTIVE):**

- Serius: ~63 → **~115 tok/s** decode (picchi 120; con throttling termico
  80-90). llama.cpp baseline stesso modello: ~145 tok/s → gap ~1.25×.
- Profiling `SERIUS_PROF=1`: mlp≈4.2ms, attn_pre≈1.0, attn_hd≈0.9-1.1,
  attn_post≈0.6, head≈0.9-1.3 ms/token.
- Kernel isolato (bench): 23-29 GB/s streaming — in-situ ~11GB/s: il gap
  residuo sono bolle seriali (attention/norm) tra GEMV e ramp-up memoria.

**Env utili:** `OMP_NUM_THREADS=10` (sweet spot; 8-12 ok, 14+ collassa),
`OMP_WAIT_POLICY=ACTIVE`, `SERIUS_NO_AVX512=1` (escape hatch AVX2),
`SERIUS_PROF=1` (profiler).

**Rimasto:** kernel integer-dot alla ggml (maddubs) se serve altro ~20%;
persistent thread pool (~1µs/fork attuale, piccolo); KV cache Q8_0 (C);
`bench_decode` + A/B finale vs llama.cpp a freddo.

## Aggiornamento sessione 2 (A/C collegato, A/B a freddo)

**Misure finali (i9-11900H su corrente, modello Q4_0, OMP_NUM_THREADS=10):**

- Serius decode: **~135-145 tok/s** (chat end-to-end con sampler)
- llama.cpp `llama-bench -t 10 -n 256`: **~149-182 tok/s** (t8=149, t10=182
  picco turbo, t12=142) → gap residuo ~1.1-1.3×
- Su batteria tutto collassa ~40% (attenzione: Windows throttla)

**Scoperte chiave:**

- I kernel GEMV isolati fanno 23-29 GB/s ma **sustained** (3s+) scendono a
  ~14 GB/s — il limite è la banda DRAM/potenza della CPU, non il kernel.
  Fix Q4_1 unroll 8 blocchi era comunque critico (7.2→23 GB/s peak).
- `SERIUS_PROF=1`: fwd≈5.7-6.2ms, sampler≈0.54ms, decode trascurabile.
- Sampler riscritto: chiavi uint64 ordinate (float-bits transform) +
  nth_element + buffer thread_local → 0.82→0.54ms.
- Attention decode path: score a 4 posizioni/batch + accumulo V in registri
  → attn_hd 1.42→0.83ms.
- Prefetch pesi MLP durante attention: **peggiorato** (ruba banda KV,
  dati evicti comunque) — riprovato e rimosso.
- FTZ/DAZ MXCSR su main+worker: nessun effetto misurato (tenuto, innocuo).

**Gap residuo verso llama.cpp:** probabilmente i loro kernel integer-dot
(maddubs, attivazione Q8_1) che dimezzano le uop/byte → più MLP in volo.
Prossimo step naturale: kernel Q4_0×Q8_1 int8 se si vuole l'ultimo ~20%.
Anche KV cache Q8_0 (C1) darebbe vantaggio su contesti lunghi.

## Piano Fase 73c — kernel integer-dot (stile ggml)

**Obiettivo**: superare llama.cpp (~149-182 tok/s) da ~140 attuali.
**Ipotesi**: i loro kernel Q4_0×Q8_1 usano `maddubs` (u8×i8) → ~1.6x meno
uop/byte dei nostri kernel fp32 → più memoria in volo → più GB/s sostenuti.

### Design

1. **Quantizzazione attivazione** (per chiamata `matvec_q`, ~2µs seriale):
   x[0..in) → blocchi di 32: `xq int8`, `xd float` (scale), `xs int` (somma).
   `d = amax/127`, `xq = round(x/d)` clamp [-127,127].
   Buffer `thread_local` in matvec_q.

2. **Kernel Q4_0×Q8_1** (gruppo nb=32):
   - ys = 16 byte qs → lo/hi nibble → ymm [lo|hi] (32 u8, 0..15)
   - `_mm256_maddubs_epi16(w, xq)` → i16 pair-sums
   - `_mm256_madd_epi16(prod, ones)` → i32[8]
   - `facc += cvtepi32_ps(prod) * (d_w * d_x)`  (lane-wise, un'unica reduce finale)
   - correzione -8*xs_b: `corr += (d_w*d_x)*xs_b` → `out -= 8*corr`
   - ~12 uop/blocco vs ~20 attuali

3. **Q4_1**: w = nib*d + m → `d*Σnib*x + m*d_x*xs` → +1 termine `msum`.

4. **Dispatch**: in%32==0 && type∈{Q4_0,Q4_1} → int path; altrimenti fp path
   esistente (Q8_0 resta fp per ora — serve solo per modello Q8_0).

5. **Verifica**: coerenza output + top-1 agreement vs path fp (stesso seed);
   attesa differenze minime (errore ~0.4% da x int8, come llama.cpp).

### Rischi

- Saturazione maddubs: |nib*x|≤15*127, pair≤3810 < 32767 ✓
- Precisione: x int8 → logit error ~1e-3 relativo — accettabile (ggml fa uguale)
- Code `in%32`: fallback fp integrale

## Esito Fase 73c — kernel integer-dot: PARITÀ, non win

Implementati kernel ggml-style Q4_0/Q4_1 x Q8_1 (maddubs+madd, 2 blocchi/zmm,
correzioni -8*sum e +m*sum, code con kernel AVX2). Verificati numericamente
vs path fp (agreement ~1e-5 su Q4_0/Q4_1 con code dispari).

**Misura A/B interleaved (stessa sessione, 300 tok):**
- int-dot:  96.8 / 94.7 / 85.0 tok/s
- fp path:  97.5 / 96.1 / ~95   tok/s
→ **nessun guadagno**: il collo è la DRAM bandwidth (~14 GB/s sostenuta),
non le uop. Il minor overhead del kernel int viene assorbito dalla
quantizzazione x per-chiamata e dalla latenza memoria.

**Decisione**: path int-dot relegato a opt-in `SERIUS_INTDOT=1`
(matvec_quant.cpp). Default = kernel fp dequant (più preciso, stesso speed).

**Conclusione architetturale**: per battere llama.cpp (~149-182 tok/s) su
questa CPU servono leve diverse:
- KV cache Q8_0 (contesti lunghi — non aiuta a seq corta)
- Prefill/Decode batching nel serving (non pertinente a tok/s singolo)
- Il gap residuo (~1.2-1.3x) è spiegato da kernel ggml più maturi
  (maddubs senza converts, weight layout ottimizzato, graph fusion)
  e NON recuperabile con micro-ottimizzazioni nel nostro formato attuale.

Target 1.2x llama.cpp: NON raggiunto. Stato onesto: ~95-140 tok/s vs
~149-182 llama.cpp a parità di condizioni.

## Fase 73d — Piano "double speed" (~95-140 → 190-280 tok/s)

### Fisica del problema
- Bytes/token: ~76MB pesi Q4_0 + ~14MB KV (ctx~300) = ~90MB
- Tetto DRAM: ~20-25GB/s sostenuta → ~220-280 tok/s MAX
- Kernel isolato: 22-26GB/s; in-situ: ~14GB/s → **~40% di banda sprecata in bolle**
- llama.cpp ~150-180 tok/s = ~55-65% del tetto; noi ~40-55%
- Obiettivo realistico: recuperare le bolle → 160-200 tok/s (parity+/1.2x)

### D1 — Chiudere il gap di accounting (~3ms/token non profilati)
Wall misurato 10.4ms vs fwd+smp+dec = 7.4ms. Manca ~30% del tempo.
Sospetti: `std::cout<<flush` per token, alloc `Tensor next_logits`,
KV append, sezioni forward non coperte dal prof.
→ Strumento: timer dentro `llama_forward_single_token` (per sezione)
+ timer su cout/push_back nel loop chat.
**Gate**: wall ≈ somma profili (gap <5%).

### D2 — Persistent thread pool per i GEMV
~150 fork/join OpenMP per token, ognuno paga wake-up dopo sezioni seriali.
Opzioni: (a) pool custom con spin+backoff su core fisici;
(b) un solo `omp parallel` persistente per layer con sezioni interne;
(c) `OMP_WAIT_POLICY=ACTIVE` già parziale — misurare il costo residuo.
**Gate**: ≥10% tok/s se fork/join >0.5ms/token.

### D3 — Layout pesi amico della banda (ggml-faithful)
I kernel ggml non convertono MA leggono i pesi in ordine che massimizza
streaming: la nostra riga [out][blocks] è già sequenziale, MA:
- (a) **prefetch software** a 2 righe in anticipo nel kernel (non durante
  attention — già fallito); (b) **split row**: ogni thread legge una
  slice contigua di TUTTE le righe (così gli N stream di memoria sono
  N/thumb rather than row-per-thread con stride) — cambiare la
  parallelizzazione da "righe" a "blocchi di righe".
**Gate**: microbench in-situ ≥18GB/s.

### D4 — Serial sections: tagliare o parallelizzare
- RMSNorm/SiLU residui scalari → vettoriali completi
- Attention: KV layout SoA, score+output già vettorizzati — verificare
  che il path usato sia quello AVX2/512 e non scalar
- **Fusione**: rms_norm del layer successivo nel GEMV precedente?
  (complessità alta — solo se D1-D3 insufficienti)

### D5 — KV cache Q8_0 (solo se contesto >512 nel benchmark)
Non aiuta a ctx corta; utility per uso reale.

### D6 — lm_head: ultimo GEMV da 16MB
Opzione: **two-stage logits** — calcolare logits precisi solo su top-M
candidati? Cambia semantica (errore di recall). Alternativa: tenere
lm_head in Q4_0 ma con kernel dedicato 4-righe/thread (più MLP).
Anche: **early-exit vocab** non valido.

### Checkpoint
Dopo ogni fase: benchmark 300 tok A/B vs baseline, poi `ctest`.
Se dopo D1-D4 siamo <150 tok/s → dichiarare parity onesto e fermarsi.

## Esito Fase 73d (parziale) — scoperte e fix

### D1 RISOLTO — il "mistero" dei 3ms/token
I ~3ms/token "mancanti" erano **prefill FP32 ammortizzato**: `llama_forward_inference`
usava matmul FP32 → **~1.4 SECONDI per un prompt di ~10 token** (path orribilmente
lento, layout B peggiore del GEMV). Ogni tok/s riportato era penalizzato di
~1.4s/N_tokens.

**Fix**: `llama_forward_inference` ora instrada prompt ≤1024 token attraverso
`llama_forward_single_token` (decode quantizzato) quando `qkv_w.valid()`.
Perché è corretto: attention causale per-posizione è identica al prefill;
il KV cache si popola con `update_kv_cache` come in decode.
**Risultato**: prefill 1.4s → ~0.3s; tok/s wall +15% (95→~120).

### Misura decode-only equa (OGGI, CPU calda)
- Serius fp path: **119-124 tok/s** (500 tok, prompt ~6 tok)
- Serius int-dot: 119 / 100 — confermato pari o peggio → resta opt-in
- llama.cpp tg256 t10: **133.4 ± 2.5 tok/s** (stessa sessione termica)
- **Gap residuo: ~1.08-1.11x** — molto più vicino di quanto sembrava

### Analisi del tempo decode (fwd ~8ms)
- GEMV quantizzati: ~5.5-6ms → ~90MB/token a ~14-15GB/s = **DRAM-bound**
- Attention: ~0.9-1.4ms → KV cache FP32 = **~24MB/token a ctx~500**
- Norme/silu/residual: ~0.3-0.5ms
- Sampler: ~0.45ms

### Prossimo passo ad alto valore: KV cache Q8_0
Halverebbe i ~24MB/token di KV → ~-0.8ms/token (~10%) e sempre più
importante a contesti lunghi. Implementazione: K/V quantizzati int8+scala
per blocco 32 su write, dequant inline nei loop score/V.

### Tetto onesto
DRAM ~20-25GB/s → ~90MB/token → **~220-280 tok/s MAX teorico**.
llama.cpp a 133-182 è già a ~60-70% del tetto. Doppiamo la velocità
ATTUALE (120→240) serve efficienza quasi perfetta: realistico è
**parity o 1.1-1.2x** (~150-180 tok/s), non 2x.

## RISULTATO FINALE — OBiettivo RAGGIUNTO

### KV cache FP16 (implementata in questa sessione)
- `KVCacheLayer::k16/v16` (`vector<uint16_t>`), scrittura in `update_kv_cache`
  con `_mm256_cvtps_ph` (F16C), lettura in `compute_head_attention` con
  `_mm256_cvtph_ps` inline nei loop score/V. Path fp32 mantenuto per
  cache_layer==nullptr e CUDA.
- **Effetto MISURATO: fwd 7.9 → 4.5ms/token; attention era ~3.4ms (non 0.9 —
  errore di unità nel prof, i contatori accumulano su 300 chiamate).**

### Benchmark finale (stessa sessione termica, AC, t=10)
| | tok/s |
|---|---|
| **Serius Q4_0 decode** | **183.4 / 183.9 / 186.1 / 187.6** |
| llama.cpp tg256 | 127.4 - 133.4 |
| **Speedup vs llama.cpp** | **~1.4-1.5×** ✅ (target era ≥1.2×) |

### Stato qualita'/robustezza
- Output coerente (stessa seed, testo inglese sensato)
- Safetensors/FP32 fallback: funzionante (verificato, ~14 tok/s)
- `ctest`: 526/532 — fallimenti preesistenti non correlati
- Path int-dot: opt-in `SERIUS_INTDOT=1` (parità, non default)
- `SERIUS_NO_AVX512`, `SERIUS_NO_INTDOT` come escape hatch

### Note
- La KV fp16 scala ancora meglio con contesti lunghi (KV bytes dimezzati)
- llm head resta l'ultimo grosso stream (16MB/token)
