# Installation & Build

Questa pagina descrive come compilare e configurare **TheSeriusProject** su CPU e, dove possibile, GPU.

---

## 1. Requisiti generali

- **C++17** o superiore.
- **CMake** ≥ 3.20.
- Sistema operativo:
  - Windows 10/11.
  - Linux recente (Ubuntu, Debian, Arch, ecc.).
  - macOS (supporto sperimentale, soprattutto lato CPU).

Opzionali:

- **CUDA Toolkit** (per backend GPU NVIDIA).
- **Python 3** per script e test avanzati.

---

## 2. Build su Windows (MSVC)

Da un terminale `x64 Native Tools Command Prompt` o da PowerShell con gli environment di Visual Studio caricati:

```bash
git clone https://github.com/<user>/TheSeriusProject.git
cd TheSeriusProject

mkdir build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Se non usi Ninja, puoi omettere `-G "Ninja"` e lasciare che CMake scelga il generatore predefinito (es. Visual Studio).

---

## 3. Build su Linux

```bash
git clone https://github.com/<user>/TheSeriusProject.git
cd TheSeriusProject

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

È consigliato abilitare l'ottimizzazione per l'architettura corrente:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCPU_NATIVE=ON
```

*(Il nome esatto dei flag dipende dalla configurazione CMake del progetto; vedere `CMakeLists.txt` per i dettagli.)*

---

## 4. Abilitare CUDA (GPU NVIDIA)

Per abilitare il backend CUDA:

1. Installa **CUDA Toolkit** compatibile con la tua GPU.
2. Assicurati che `nvcc` sia nel `PATH`.
3. Rigenera la build:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build . --config Release
```

I flag precisi (`ENABLE_CUDA`, `CUDA_ARCH`, ecc.) possono variare; controlla i relativi `CMakeLists.txt` in `src/blas/` e `src/hal/`.

---

## 5. Layout di output

Dopo la build, il binario principale e le librerie vengono prodotti in `build/` (o nella cartella configurata dal generatore).

File utili nella root del progetto:

- `build_log.txt`, `build_output.txt`, `build_last_error.txt` — log di build e errori recenti.
- `*_bench.txt` — risultati dei benchmark eseguiti.

---

## 6. Verifica installazione

Esegui dalla cartella `build`:

```bash
ctest --output-on-failure
```

Se tutti i test principali passano (autograd, BLAS, tensor, transformer, training), l'installazione di base è corretta.

---

## 7. Prossimi passi

- Vai su `docs/getting-started.md` per un percorso guidato.
- Consulta `docs/gpu-setup.md` (quando disponibile) per dettagli avanzati sul setup GPU.
- Leggi `docs/supported-models.md` per capire quali modelli sono target del progetto.

