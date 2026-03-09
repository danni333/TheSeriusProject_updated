# Getting Started

Benvenuto in **TheSeriusProject**.  
Questa guida ti porta da zero a eseguire test e benchmark di base in pochi minuti.

---

## 1. Prerequisiti

- Compilatore **C++17** o superiore (MSVC, Clang, GCC).
- **CMake** ≥ 3.20.
- (Opzionale) **CUDA Toolkit** per il backend GPU.
- (Opzionale) **Python 3** per script di supporto e benchmark.

Su Windows è consigliato:

- Visual Studio con workload "Desktop development with C++".
- `CMake` installato e disponibile nel `PATH`.

---

## 2. Clona il repository

```bash
git clone https://github.com/<user>/TheSeriusProject.git
cd TheSeriusProject
```

Se stai lavorando da una copia zip, assicurati che il contenuto sia estratto in una cartella senza spazi nel percorso (per evitare problemi con alcuni tool di build).

---

## 3. Build veloce (CPU only)

Da root del progetto:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Se la build fallisce, controlla i file `build_*.txt` generati nella root (es. `build_log.txt`, `build_last_error.txt`) per i dettagli.

---

## 4. Esegui i test principali

Sempre dalla cartella `build`:

```bash
ctest --output-on-failure
```

Questo esegue la suite di test definita in `tests/`, inclusi:

- Autograd e engine tensoriale.
- Operazioni BLAS (matmul, ecc.).
- Kernel CUDA (se compilati).
- Transformer stack (attenzione, FFN, KV cache, ecc.).

---

## 5. Esegui i benchmark

Per avere un'idea delle performance di base:

```bash
ctest -R bench -V
```

I risultati dettagliati vengono spesso riportati in file come:

- `matmul_bench.txt`
- `memory_bench.txt`
- `mixed_precision_bench.txt`
- `bench_results.txt`

Controlla questi file nella root del progetto per analizzare le performance.

---

## 6. Prossimi passi

- Leggi `docs/installation.md` per una configurazione più approfondita (GPU, flag di compilazione, ecc.).
- Esplora `docs/architecture.md` per capire come è organizzato il motore interno.
- Segui la roadmap in `todo/` per vedere le fasi previste (quantizzazione, distribuzione, API, documentazione avanzata, ecc.).

Se incontri problemi, consulta `docs/troubleshooting.md` o apri una issue nel repository.

