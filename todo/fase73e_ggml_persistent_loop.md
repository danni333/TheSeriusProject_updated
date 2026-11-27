# Fase 73e — Loop persistente stile ggml (thread pool spin + barrier a senso)

**Obiettivo**: avvicinare il decode CPU a llama.cpp (~149-182 tok/s vs nostri
~95-140). Ipotesi documentata in `fase73_inferenza_rapida.md` (Fase 73d):
ogni `matvec_q` apre una regione `#pragma omp parallel for` — decine di
regioni per token, ognuna paga wake-up dei thread (~5-30µs) + barrier OpenMP.
ggml usa worker persistenti che **spin-wait** su contatori atomici: dispatch
~1-2µs e niente handoff al kernel scheduler.

**Vincolo metodologico** (metodo scientifico): ogni fase ha un gate di
misura — se il guadagno atteso non si materializza, ci si ferma e si
documenta (come Fase 73c).

## Fase A — Misura del costo attuale (niente modifiche al codice caldo)

| # | Task | File | Verifica |
|---|------|------|----------|
| A1 | Microbenchmark: costo di una regione `#pragma omp parallel for` vuota (loop 0 iters) con OMP_NUM_THREADS=8/16, best-of-1000 | `tests/bench_omp_region.cpp` (nuovo, ~40 righe) | numero µs/region |
| A2 | Contare le regioni parallele per token nel decode: `matvec_q` ×4/layer + attention heads + rms_norm — stimare #region/token per 7B (32 layer → ~130+) | `src/infer/llama_inference.cpp` (solo lettura) | tabella layer→regioni |
| A3 | **Gate decisionale**: se costo_REGION × regioni/token < ~3% del tempo/token → loop persistente NON conviene, documentare e chiudere | `todo/fase73e_*.md` | decisione scritta |

## Fase B — Pool persistente autonomo (testabile da solo)

| # | Task | File | Note |
|---|------|------|------|
| B1 | `class SpinPool`: N-1 worker persistenti + caller esegue chunk 0 (come ggml thread 0). Creato lazy al primo use, mai distrutto fino a exit | `src/common/spin_pool.h/.cpp` | nuovo, ~150 righe |
| B2 | Dispatch a ticket atomici: `parallel_for(n_items, grain, fn)` → contatore `job_id` (sense-reversing), ogni worker prende chunk `ticket++` | `spin_pool.cpp` | niente code, niente mutex nel fast path |
| B3 | Spin policy: `_mm_pause()` loop; dopo ~1ms senza lavoro → `Sleep(0)`/yield; flag `active` per tornare a spin durante decode | `spin_pool.cpp` | ev`. busy-wait solo quando serve |
| B4 | Test correttezza: somma/risultati identici a loop seriale per n dispari, grain 1 e grain>1, chiamate concorrenti da 2 thread (raro ma le serializziamo con mutex esterno) | `tests/test_spin_pool.cpp` | gtest |
| B5 | Microbench dispatch: `parallel_for` vuoto vs `#pragma omp parallel for` vuoto (stesso banco di A1) | `bench_omp_region.cpp` esteso | **Gate B**: dispatch < ~30% del costo OMP altrimenti stop |

## Fase C — Integrazione in `matvec_q` (primo consumatore reale)

| # | Task | File | Note |
|---|------|------|------|
| C1 | `matvec_q`: se `SpinPool` attivo e `N >= 256` → `pool.parallel_for` su chunk di righe invece di `#pragma omp parallel for`; `xq`/`XQuant` già read-only condiviso | `src/blas/matvec_quant.cpp` | dietro env `SERIUS_SPINPOOL` (default OFF inizialmente) |
| C2 | Idem per il path a due righe `q40_row2_avx512` (chunk = paia di righe) e per l'int-dot path | `matvec_q` | stesso env gate |
| C3 | `prefetch_quant_weight`: stessa sostituzione del pragma | `matvec_quant.cpp` | |
| C4 | A/B: `bench_matvec_quant.exe` + `serius bench` con `SERIUS_SPINPOOL=1` vs `=0`, best-of, a sistema scarico | — | **Gate C**: ≥5% decode altrimenti default resta OMP e si documenta |

## Fase D — Decode-level (solo se C conferma)

| # | Task | File | Note |
|---|------|------|------|
| D1 | `compute_head_attention`: `#pragma omp parallel for` su teste → `SpinPool` se attivo | `llama_inference.cpp` | teste = work items granuali |
| D2 | `rms_norm` / residual adds: valutare se il loop seriale basta o se vale pool per seq>1 | `llama_inference.cpp` | solo misurando |
| D3 | (opzionale) Fondere le 4 `matvec_q` di una layer in un'unica region se i pesi sono indipendenti — gate_up e down no (dipendenza), ma qkv+o no (attention in mezzo) → **probabilmente NO**, documentare | — | analisi, non codice |
| D4 | Idle policy: pool va in sleep tra token (generate ha gap) — flag `begin_decode()/end_decode()` attorno al loop di generazione | `spin_pool`, `llama_inference.cpp` | evita 7 core bruciati per idle |

## Fase E — Pulizia e documentazione

| # | Task |
|---|------|
| E1 | Default ON solo se A+B+C danno ≥10% tok/s misurato; altrimenti env-only e documentato come "parity" |
| E2 | Aggiornare `BENCHMARKS.md` + `todo/fase73_inferenza_rapida.md` con esiti |
| E3 | `ctest` completo + `bench_matvec_quant` finale |

## Rischi noti

- **Busy-wait vs termica**: 7 worker che spinnano scaldano — mitigato da
  sleep dopo idle timeout e da `begin/end_decode`.
- **Priorità thread**: worker a priorità normale possono essere preempted
  → code path di fallback a OMP se un dispatch supera timeout (opzionale,
  Fase D).
- **Numeri attesi (onesti)**: se OMP wake ≈ 10µs e ~130 regioni/token →
  ~1.3ms/token; a ~10ms/token attuale ≈ **max ~13%**. Se wake ≈ 3µs → ~4%.
  Il vero gap verso llama.cpp (~1.3x) NON si chiude qui — questo recupera
  solo le "bolle" di sincronizzazione stimate ~40% di banda sprecata
  (vedi Fase 73d); servono entrambe le cose.
- **Interazione OMP**: se `OMP_NUM_THREADS` > core fisici o altre lib
  (OpenMP di test) — il pool usa `hardware_concurrency` o env dedicato
  `SERIUS_SPIN_THREADS`.

## Primo task concreto (A1)

Creare `tests/bench_omp_region.cpp`: misura best/median di N regioni
`#pragma omp parallel for` vuote a 8 e 16 thread → output µs/region.
~30 minuti di lavoro, zero rischio, e decide se il progetto è worthwhile.

## Esito Fase 73e — PARITÀ, non win (2026-11-27)

**Implementato**: `src/common/spin_pool.{h,cpp}` — pool persistente
ggml-style (worker 1..N-1 + caller=chunk 0, epoch atomico + contatore
`remaining`, spin `_mm_pause`/`Sleep(0)` dopo 200k iter, ready-barrier
nel costruttore, mutex di dispatch per server concorrenti, distruttore
senza epoch-bump per non eseguire job con riferimenti dangling).
`tests/test_spin_pool.cpp` — 5 test verdi. Integrazione dietro
`SERIUS_SPINPOOL=1` (default OFF): `matvec_q` (int-dot, q40 pairs, fp,
prefetch) + attention heads in `llama_inference.cpp` (D1 anticipata:
necessaria per evitare oversubscription OMP+spin).

**Misure**:
- A1: dispatch OMP ~0.8-1.6µs/region @8 thread (caldo), collassa a 16.
- Dispatch SpinPool: ~0.6µs (10000 dispatch in 6ms, smoke test).
- E2E decode CPU (Qwen2.5-0.5B Q4_K_M, 96 tok, OMP_NUM_THREADS=8):
  SPINPOOL=1 → 29.4/30.2/30.0 tok/s; SPINPOOL=0 → 29.5/28.5/30.4 tok/s.
  **Delta: ~0% (parità)**.
- Integrazione parziale (solo matvec, attention ancora OMP):
  ~28.3 tok/s → **−19%** per oversubscription 7 spinner + 8 thread OMP.
  Lezione: se si adotta, TUTTO il parallelismo decode deve passare dal
  pool, non solo i GEMV.

**Perché non vince**: MSVC OpenMP è già ~1.6µs/region a caldo e il default
`OMP_WAIT_POLICY=ACTIVE` tiene i worker in spin tra regioni consecutive
(i gap seriali del decode sono ~µs, sotto la soglia di sleep). Il
persistente recupera solo ~1µs/region × ~130 regioni ≈ 0.13ms/token —
sotto il rumore e sotto il gate C (≥5%).

**Decisione**: codice mantenuto dietro env flag per eventuale retest su
altri OS (su Linux GOMP ha wake più costoso — lì potrebbe vincere).
Default resta OMP. Prossimo gap reale verso llama.cpp: altrove
(kernel layout, KV quant, graph fusion) — vedi fase73_inferenza_rapida.
