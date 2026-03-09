# Troubleshooting

Questa pagina raccoglie problemi comuni e relative soluzioni quando si lavora con **TheSeriusProject**.

---

## 1. Problemi di build

### 1.1 CMake non trova il compilatore

**Sintomi**

- Errori del tipo: `No CMAKE_CXX_COMPILER could be found`.

**Possibili cause**

- Toolchain C++ non installata o non configurata nel `PATH`.

**Soluzione**

- Su Windows: installa Visual Studio con il workload "Desktop development with C++".
- Su Linux: installa `build-essential` (Debian/Ubuntu) o pacchetto equivalente.

---

### 1.2 Errori legati a CUDA

**Sintomi**

- Errori durante la configurazione CMake o la compilazione di file `.cu`.

**Possibili cause**

- CUDA Toolkit non installato o versione incompatibile.
- Variabili d'ambiente (`CUDA_HOME`, `PATH`) non impostate correttamente.

**Soluzione**

- Verifica che `nvcc --version` funzioni nel terminale.
- Ricompila con `-DENABLE_CUDA=OFF` per isolare il problema e lavorare solo su CPU.

---

## 2. Test che falliscono

### 2.1 Test numerici (BLAS, autograd, transformer) falliscono marginalmente

**Sintomi**

- Asserzioni del tipo: "expected ~X, got Y (difference > epsilon)".

**Possibili cause**

- Differenze tra implementazioni SIMD/GPU e scalar.
- Differenze di floating point tra piattaforme/compilatori.

**Soluzione (linea guida)**

- Verifica la tolleranza numerica usata nel test.
- Confronta i risultati con la versione scalar di riferimento (se disponibile).

---

### 2.2 Benchmark estremamente lenti

**Sintomi**

- Tempi di `matmul` o altri benchmark molto superiori alle aspettative.

**Possibili cause**

- Mancato utilizzo di AVX/AVX2/AVX‑512.
- Compilatore che non sta ottimizzando correttamente (es. build `Debug`).

**Soluzione**

- Ricompila con `-DCMAKE_BUILD_TYPE=Release`.
- Controlla i flag specifici per abilitare il path SIMD (vedi `src/blas/dispatch.cpp` e `CMakeLists.txt`).

---

## 3. Altri problemi

Se incontri un problema non documentato:

1. Controlla i file di log nella root (`build_log.txt`, `build_last_error.txt`, `*_bench.txt`, `test_output*.txt`).
2. Riprova con una build pulita (`rm -rf build && mkdir build`).
3. Apri una issue nel repository includendo:
   - Sistema operativo e versione.
   - Versione del compilatore/CMake/CUDA.
   - Log di build o di esecuzione rilevanti.

Questa pagina verrà estesa con casi sempre più specifici man mano che il progetto evolve.

