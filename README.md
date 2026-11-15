# TheSeriusProject

Progetto C++ per un motore di inferenza LLM locale, scritto in C++, con supporto per accelerazione opzionale (CUDA, MPS, ecc.).

Licenza: Apache 2.0

> Obiettivo: costruire un motore matematico locale, riproducibile e didattico, focalizzato su performance, trasparenza e controllo totale sul ciclo di vita del modello.

---

## Panoramica

TheSeriusProject fornisce:

- **Motore tensoriale custom**: implementazione di operazioni base (BLAS‑like), autograd e kernel specializzati.
- **Supporto multi‑backend**: CPU scalar/AVX2/AVX‑512, CUDA (dove disponibile), FP16/BF16.
- **Stack Transformer completo**: attenzione (GQA, MLA-like), RoPE, FFN, RMSNorm, KV cache, MoE.
- **Inferenza LLaMA/GGUF**: loader GGUF (Q4_0/Q4_1/Q8_0/Q4_K/Q6_K), pesi quantizzati packed per il decode, conversione GGUF→safetensors.
- **Server HTTP OpenAI‑compatibile**: chat/completions, completions, embeddings, streaming SSE, multi-modello a runtime (load/unload, `hf://`).
- **Web UI integrata**: chat browser servita direttamente dal binario C++ (`ui/`).
- **Calcolo Distribuito**: supporto per Data Parallelism (DP), Pipeline Parallelism (PP) e Tensor Parallelism (TP).
- **Suite di test & benchmark**: test unitari granulari e benchmark per matmul, memoria, mixed precision, ecc.

Il progetto è organizzato in più fasi (`todo/`), ciascuna focalizzata su un aspetto (fondamenta, tokenizer, quantizzazione, distribuzione, documentazione, ecc.).

---

## Stato del progetto

- **Fase corrente**: espansione funzionalità core (autograd, BLAS, transformer, training) e miglioramento della copertura test.
- **Stabilità**: API C++ ancora soggette a cambiamenti; l'interfaccia CLI/API esterna è in fase di definizione.
- **Target**: sviluppatori interessati a capire e sperimentare con l'implementazione di un LLM engine from‑scratch.

Per dettagli sul piano a lungo termine vedere i file in `todo/` (in particolare `todo/fase0_fondamenta.md` e `todo/fase51_documentazione.md`).

---

## Quick Start (da sorgente)

> Nota: i binari precompilati non sono ancora considerati stabili; attualmente il flusso raccomandato è la compilazione da sorgente.

### Requisiti

- **Compilatore C++20** (MSVC 2022, Clang, GCC).
- **CMake** (>= 3.25).
- **Python** opzionale per script di test/benchmark.
- **CUDA Toolkit** opzionale per abilitare il backend GPU.

### Clona il repository

```bash
git clone https://github.com/danni333/TheSeriusProject_updated.git
cd TheSeriusProject_updated
```

### Compilazione base (CPU)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Su Windows con MSVC serve l'ambiente developer (Developer PowerShell o `Launch-VsDevShell.ps1`). Il binario finisce in `build/src/serius` (o `build-ninja/src/serius.exe` con Ninja).

L'output di build e gli eventuali errori vengono registrati nei file `build_*.txt` nella root del progetto.

### Configurazione MoE e prova locale

Il loader JSON supporta sia il blocco annidato `moe` sia i campi flat `moe_enabled`, `num_experts`, `top_k`, ecc. Per provare il percorso completo di inferenza su CPU usa il modello dimostrativo piccolo:

```bash
./build/src/serius chat --config configs/model_moe_demo.json --type transformer --max-new-tokens 16
```

`--max-new-tokens` limita la generazione per prove rapide. `configs/model_moe_1b.json` è invece un profilo molto più grande e richiede memoria significativamente maggiore.
Il profilo demo usa pesi inizializzati casualmente: serve a verificare parsing, routing MoE e inferenza end-to-end, non a produrre risposte linguistiche utili.

### Server HTTP + Web UI

```bash
# Modello locale (dir safetensors, config.json o file .gguf) oppure hf://repo
./build/src/serius serve -m models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf --type llama -p 8080 --ui ui

# Poi apri http://localhost:8080 nel browser.
```

- **Multi-modello a runtime**: `POST /v1/models/load` e `DELETE /v1/models/{id}` permettono di caricare/scaricare modelli mentre il server gira; la UI ha un pannello Models integrato. I load girano in parallelo all'inference.
- **KV cache persistente**: in chat multi-turn il server riusa il KV cache dei turni precedenti quando il prompt è un'estensione esatta (stesso modello e system prompt).
- **Env utili**: `SERIUS_PREFILL_BATCH=1` abilita il prefill batched (default: token-per-token, più lento ma identico al decode), `SERIUS_KV_FP32=1` forsa KV cache FP32, `SERIUS_GPU_TIME=1` logga i tempi GPU.

### Esecuzione test

All'interno della cartella di build:

```bash
ctest --output-on-failure
```

I sorgenti dei test si trovano in `tests/` (autograd, BLAS, CUDA, tensor, transformer, training, ecc.).

---

## Struttura del codice (alto livello)

- `src/autograd/` — motore di autograd e primitive di calcolo.
- `src/blas/` — implementazioni di `matmul` e altre operazioni a basso livello (scalar, AVX2, AVX512, CUDA, quantized GEMV).
- `src/tensor/` — implementazione del tipo `Tensor` e logica di gestione memoria (host/device sync).
- `src/memory/` — allocator e gestione pool di memoria.
- `src/transformer/` — blocchi transformer (attenzione, FFN, layernorm, KV cache, modello LLaMA completo).
- `src/infer/` — inferenza LLaMA (prefill, decode, KV cache, sampler).
- `src/io/` — loader modelli: GGUF, safetensors, config JSON, conversione.
- `src/tokenizer/` — tokenizer BPE con metadata GGUF (BOS/EOS).
- `src/server/` — server HTTP (endpoint OpenAI-compat, registry multi-modello, static files per la UI).
- `src/cli/` — CLI (`chat`, `serve`, `generate`, `convert`, `import`, `doctor`, `bench`, `stress`).
- `src/train/` — training loop, checkpointing, ottimizzatori (es. AdamW).
- `src/quant/`, `src/attention/`, `src/hal/`, `src/distributed/`, `src/hub/` — quantizzazione, flash attention, astrazione device, parallelismo, model hub.
- `ui/` — web UI (HTML/JS/CSS) servita dal server.
- `tests/` — test unitari e benchmark.
- `todo/` — roadmap dettagliata per le varie fasi del progetto.

Per una descrizione più approfondita dell'architettura vedi `docs/architecture.md`.

---

## Documentazione

La documentazione è in corso di scrittura nella cartella `docs/`:

- `docs/getting-started.md` — guida rapida per partire in pochi minuti.
- `docs/installation.md` — dettagli su compilazione, dipendenze e setup GPU.
- `docs/cli-reference.md` — riferimento completo (in costruzione) per i comandi CLI.
- `docs/api-reference.md` — reference delle API (REST/OpenAI‑compat, quando disponibili).
- `docs/configuration.md` — formato dei file di configurazione e parametri principali.
- `docs/supported-models.md` — panoramica sui modelli supportati e requisiti.
- `docs/troubleshooting.md` — problemi comuni e relative soluzioni.
- `docs/architecture.md` — descrizione dell'architettura interna per sviluppatori.
- `docs/changelog.md` — cronologia delle modifiche principali.

Se un file è vuoto o marcato come "WIP", significa che la sezione è pianificata ma non ancora finalizzata.

---

## Come contribuire

1. Fai un fork del repository.
2. Crea un branch tematico: `feature/...` o `fix/...`.
3. Assicurati che i test esistenti passino e, se possibile, aggiungi test per le nuove funzionalità.
4. Invia una pull request descrivendo:
   - Il **contesto** del cambiamento.
   - Il **comportamento atteso**.
   - Eventuali **limitazioni** o TODO rimasti.

Linee guida più dettagliate sono disponibili in `docs/contributing.md` (in aggiornamento).

---

## Licenza

Questo progetto è distribuito sotto licenza **Apache 2.0**. Vedi il file `LICENSE` (quando presente) per i dettagli completi.
