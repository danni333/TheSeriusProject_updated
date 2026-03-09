# TheSeriusProject

Progetto C++ per un motore di inferenza LLM locale, scritto in C++, con supporto per accelerazione opzionale (CUDA, MPS, ecc.).

Licenza: Apache 2.0

> Obiettivo: costruire un motore matematico locale, riproducibile e didattico, focalizzato su performance, trasparenza e controllo totale sul ciclo di vita del modello.

---

## Panoramica

TheSeriusProject fornisce:

- **Motore tensoriale custom**: implementazione di operazioni base (BLAS‑like), autograd e kernel specializzati.
- **Supporto multi‑backend**: CPU scalar/AVX/AVX2/AVX‑512, CUDA (dove disponibile).
- **Stack Transformer completo**: attenzione, FFN, layernorm, KV cache, positional encoding, training loop.
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

- **Compilatore C++17** o superiore (MSVC, Clang, GCC).
- **CMake** (>= 3.20 consigliato).
- **Python** opzionale per script di test/benchmark.
- **CUDA Toolkit** opzionale per abilitare il backend GPU.

### Clona il repository

```bash
git clone https://github.com/<user>/TheSeriusProject.git
cd TheSeriusProject
```

### Compilazione base (CPU)

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

L'output di build e gli eventuali errori vengono registrati nei file `build_*.txt` nella root del progetto.

### Esecuzione test

All'interno della cartella `build`:

```bash
ctest --output-on-failure
```

I sorgenti dei test si trovano in `tests/` (autograd, BLAS, CUDA, tensor, transformer, training, ecc.).

---

## Struttura del codice (alto livello)

- `src/autograd/` — motore di autograd e primitive di calcolo.
- `src/blas/` — implementazioni di `matmul` e altre operazioni a basso livello (scalar, AVX, CUDA).
- `src/tensor/` — implementazione del tipo `Tensor` e logica di gestione memoria.
- `src/memory/` — allocator e gestione pool di memoria.
- `src/transformer/` — blocchi transformer (attenzione, FFN, layernorm, KV cache, modello completo).
- `src/train/` — training loop, checkpointing, ottimizzatori (es. AdamW).
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
