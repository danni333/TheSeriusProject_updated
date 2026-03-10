# Installazione — TheSeriusProject

Istruzioni per compilare e installare il progetto da sorgente.

---

## Requisiti di sistema

| Requisito | Minimo |
|-----------|--------|
| Compilatore C++ | C++20 (MSVC 2022, GCC 12+, Clang 16+) |
| CMake | 3.25 o superiore |
| Git | Per clonare e per FetchContent |
| RAM | 8 GB (16+ GB per modelli grandi) |

Opzionali:

- **CUDA Toolkit 12.x** — per backend GPU NVIDIA
- **Vulkan SDK** — per backend Vulkan
- **Xcode / Metal** — per backend Metal su macOS
- **Python 3** — per script (generate_reference.py, download_model.py)
- **NumPy** — per generare dati di test (script)

---

## Build su Windows

1. Apri **Developer Command Prompt for VS** (o terminale con `cl` in PATH).

2. Dalla root del progetto:

```bat
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

3. Con CUDA:

```bat
cmake .. -DCMAKE_BUILD_TYPE=Release -DSERIUS_USE_CUDA=ON
cmake --build . --config Release
```

4. Eseguibile CLI: `build\Release\serius.exe` (o `build\serius.exe` a seconda del generatore).

---

## Build su Linux / macOS

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Con Ninja:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## Test

Dalla cartella `build`:

```bash
ctest --output-on-failure
```

Oppure eseguire un singolo test:

```bash
./tests/test_tensor
./tests/test_sampler
```

---

## Configurazione modelli

Dopo la build, i file in `configs/` possono essere usati senza installazione:

- `configs/model_1b.json` — modello piccolo (demo)
- `configs/model_7b.json` — modello 7B (parametri)
- `configs/training_config.json` — training

Per usare un config in un’altra cartella, passa il path assoluto o relativo a `serius chat --config path/to/config.json`.

---

## Installazione (opzionale)

CMake non definisce ancora un target `install` completo. Per usare il progetto:

- Esegui `serius` dalla cartella `build` (o aggiungi `build/` al PATH).
- Copia manualmente `serius` e la cartella `configs/` dove preferisci.

In futuro: `cmake --build build --target install` con prefisso configurabile.
