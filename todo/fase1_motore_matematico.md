# Fase 1 — Il Motore Matematico (Tensori e BLAS)

> Un LLM è, in essenza, una gigantesca serie di moltiplicazioni di matrici.
> Prima di poter addestrare o eseguire inference su qualsiasi modello, serve un'infrastruttura
> solida per gestire dati multidimensionali, eseguire algebra lineare ad alta velocità
> e gestire la memoria in modo efficiente.

---

## 1. Classe `Tensor`

### Obiettivo

Creare una classe C++ che rappresenti un **array multidimensionale generico** (tensore), con allocazione **contigua** in memoria per massimizzare le performance della cache.

### Struttura dati interna

| Campo | Tipo | Descrizione |
|---|---|---|
| `data_` | `float*` | Puntatore al blocco contiguo di memoria |
| `shape_` | `std::vector<size_t>` | Dimensioni del tensore (es. `{batch, rows, cols}`) |
| `strides_` | `std::vector<size_t>` | Offset per accesso a ciascuna dimensione |
| `size_` | `size_t` | Numero totale di elementi |
| `owns_data_` | `bool` | Se `true`, il distruttore libera la memoria |

### Operazioni fondamentali

```
Tensor(shape)                  → Alloca memoria contigua
Tensor(data_ptr, shape)        → View su memoria esterna (non possiede i dati)
operator()(i, j, ...)          → Accesso indicizzato con strides
reshape(new_shape)             → Cambia forma senza copiare (se compatibile)
slice(dim, start, end)         → Restituisce una view su un sotto-tensore
fill(value)                    → Riempie con un valore
copy()                         → Deep copy
```

### Dettagli tecnici chiave

- **Layout Row-Major (C-order)**: gli elementi di una riga sono contigui in memoria → ottimale per cache L1/L2.
- **Strides calcolati automaticamente** dalla shape:

  ```
  strides[i] = strides[i+1] * shape[i+1]
  strides[last] = 1
  ```

- **View vs. Copy**: un tensore può essere una *view* su un altro (condivide il buffer).
  Questo è critico per le operazioni di slicing senza overhead.

### File previsti

| File | Contenuto |
|---|---|
| `src/tensor/tensor.h` | Dichiarazione della classe `Tensor` |
| `src/tensor/tensor.cpp` | Implementazione |
| `tests/test_tensor.cpp` | Unit test |

---

## 2. Algebra Lineare (BLAS-like)

### Obiettivo

Implementare le operazioni di algebra lineare necessarie per un LLM, a partire dalla **moltiplicazione di matrici** (`C = A × B`), ottimizzata con istruzioni **SIMD**.

### Operazioni da implementare

| Operazione | Formula | Priorità |
|---|---|---|
| `matmul` | $C_{ij} = \sum_k A_{ik} \cdot B_{kj}$ | 🔴 Critica |
| `matvec` | $y_i = \sum_j A_{ij} \cdot x_j$ | 🟡 Alta |
| `element_wise_add` | $C_{ij} = A_{ij} + B_{ij}$ | 🟢 Media |
| `element_wise_mul` | $C_{ij} = A_{ij} \cdot B_{ij}$ | 🟢 Media |
| `transpose` | $B_{ij} = A_{ji}$ | 🟡 Alta |
| `scale` | $B_{ij} = \alpha \cdot A_{ij}$ | 🟢 Media |

### Ottimizzazione SIMD (AVX2 / AVX-512 / AMX)

Le istruzioni SIMD (Single Instruction, Multiple Data) permettono di operare su **8 float** (AVX2), **16 float** (AVX-512), oppure intere **tile di matrici** (Intel AMX su Sapphire Rapids+) in un singolo o pochi cicli di clock.

#### Strategia di implementazione

```
┌─────────────────────────────────────────────┐
│              matmul(A, B, C)                │
│                                             │
│  1. Verifica allineamento memoria (32/64B)  │
│  2. Loop esterno: righe di A               │
│  3. Loop medio: colonne di B (tiled)       │
│  4. Loop interno: prodotto scalare con SIMD │
│     ├── _mm256_loadu_ps()   (carica 8 float)│
│     ├── _mm256_fmadd_ps()   (fused mul-add) │
│     └── _mm256_storeu_ps()  (salva 8 float) │
│  5. Gestisci il "tail" (elementi rimanenti) │
└─────────────────────────────────────────────┘
```

#### Tiling (blocchi di cache)

Per matrici grandi, la matmul naive ha pessime performance di cache. Si usa il **tiling**:

```
Per blocchi di dimensione TILE (tipicamente 64-128):
  Per ogni blocco di righe di A (i):
    Per ogni blocco di colonne di B (j):
      Per ogni blocco della dimensione interna (k):
        micro_kernel(A_block, B_block, C_block)  ← SIMD qui
```

> [!IMPORTANT]
> Il **tiling size** deve essere scelto in base alla dimensione della cache L1
> (tipicamente 32 KB). Un tile di 64×64 float = 16 KB, che sta comodamente in L1.

#### Fallback

Se la CPU non supporta AVX2, fornire un fallback con loop scalare.
Usare `__builtin_cpu_supports("avx2")` per il runtime dispatch.

### File previsti

| File | Contenuto |
|---|---|
| `src/blas/blas.h` | Dichiarazioni delle operazioni |
| `src/blas/matmul_avx2.cpp` | Implementazione matmul con AVX2 |
| `src/blas/matmul_avx512.cpp` | Implementazione matmul con AVX-512 |
| `src/blas/matmul_scalar.cpp` | Fallback scalare |
| `src/blas/elementwise.cpp` | Operazioni element-wise |
| `src/blas/dispatch.cpp` | Runtime CPU dispatch |
| `tests/test_blas.cpp` | Unit test e benchmark |

---

## 3. Gestione Memoria — Memory Pooling

### Obiettivo

Evitare allocazioni/deallocazioni frequenti (`new`/`delete`, `malloc`/`free`) durante il training. Le chiamate ripetute al sistema operativo:

- **Frammentano la RAM** (tanti piccoli blocchi sparsi)
- **Sono lente** (syscall overhead)
- **Distruggono la località della cache**

### Architettura: Arena Allocator

```
┌──────────────────────────────────────────────────────┐
│                    MemoryPool                        │
│                                                      │
│  ┌─────────────┬─────────────┬──────────────────┐   │
│  │  Blocco 1   │  Blocco 2   │  Blocco 3 (free) │   │
│  │  (in uso)   │  (in uso)   │                   │   │
│  └─────────────┴─────────────┴──────────────────┘   │
│  ▲                            ▲                      │
│  base_ptr_                    offset_                │
│                                                      │
│  allocate(size):                                     │
│    ptr = base_ptr_ + offset_                         │
│    offset_ += align(size)                            │
│    return ptr                                        │
│                                                      │
│  reset():                                            │
│    offset_ = 0   ← Resetta tutto in O(1)             │
└──────────────────────────────────────────────────────┘
```

### Design pattern

| Componente | Responsabilità |
|---|---|
| `MemoryPool` | Alloca un grande blocco iniziale, distribuisce fette allineate |
| `PoolAllocator<T>` | Wrapper STL-compatible per usare il pool con `std::vector` ecc. |
| `ScopedArena` | RAII wrapper — resetta automaticamente il pool a fine scope |

### Dettagli tecnici

- **Allineamento**: ogni allocazione è allineata a **64 byte** (cache line) per evitare false sharing e per compatibilità SIMD.
- **Strategia di crescita**: se il pool si riempie, alloca un nuovo blocco (linked-list di arene) oppure raddoppia (`grow()`).
- **Thread safety**: per il momento single-threaded. In futuro si può aggiungere un pool per-thread.
- **Integrazione con Tensor**: la classe `Tensor` accetta un `MemoryPool*` opzionale nel costruttore. Se fornito, alloca dal pool anziché dall'heap.

### File previsti

| File | Contenuto |
|---|---|
| `src/memory/memory_pool.h` | Dichiarazione `MemoryPool`, `ScopedArena` |
| `src/memory/memory_pool.cpp` | Implementazione |
| `src/memory/pool_allocator.h` | `PoolAllocator<T>` STL-compatible |
| `tests/test_memory.cpp` | Unit test (leak check, alignment) |

---

## 4. Struttura del Progetto (Fase 1)

```
TheSeriusProject/
├── CMakeLists.txt              ← Build system
├── src/
│   ├── tensor/
│   │   ├── tensor.h
│   │   └── tensor.cpp
│   ├── blas/
│   │   ├── blas.h
│   │   ├── matmul_avx2.cpp
│   │   ├── matmul_avx512.cpp
│   │   ├── matmul_scalar.cpp
│   │   ├── elementwise.cpp
│   │   └── dispatch.cpp
│   └── memory/
│       ├── memory_pool.h
│       ├── memory_pool.cpp
│       └── pool_allocator.h
├── tests/
│   ├── test_tensor.cpp
│   ├── test_blas.cpp
│   └── test_memory.cpp
└── README.md
```

---

## 5. Piano di Verifica

### Test Automatici

| Test | Cosa verifica | Comando |
|---|---|---|
| `test_tensor` | Creazione, reshape, slice, view vs copy, accesso indicizzato | `ctest --test-dir build -R test_tensor` |
| `test_blas` | Correttezza matmul (confronto con risultato naive), elementwise ops | `ctest --test-dir build -R test_blas` |
| `test_memory` | Allocazione dal pool, allineamento a 64B, reset, crescita pool | `ctest --test-dir build -R test_memory` |

### Benchmark

| Benchmark | Cosa misura |
|---|---|
| `bench_matmul` | Tempo matmul SIMD vs naive per matrici 256×256, 512×512, 1024×1024 |
| `bench_memory` | Throughput allocazione pool vs `malloc` |

### Build e Run

```bash
# Configurazione
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Esecuzione test
ctest --test-dir build --output-on-failure

# Benchmark (opzionale)
./build/bench_matmul
./build/bench_memory
```

---

## 6. Dipendenze e Requisiti

| Requisito | Minimo | Raccomandato |
|---|---|---|
| Compilatore C++ | C++17 | C++20 |
| CMake | 3.20 | 3.28+ |
| CPU SIMD | SSE4.2 | AVX-512 o Intel AMX |
| RAM | 8 GB | 16+ GB |
| OS | Windows 10/11 | Windows 11 |

> [!NOTE]
> Per la Fase 1 non servono librerie esterne. Tutto è implementato da zero.
> In fasi successive si potrà opzionalmente interfacciare con OpenBLAS o MKL per confronto.
