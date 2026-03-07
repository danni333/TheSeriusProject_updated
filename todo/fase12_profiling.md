# Fase 12 — Profiling Avanzato (NSight Systems/Compute, Tracy, Roofline Analysis)

> In C++ non hai i messaggi di errore chiari di Python.
> Se un peso diventa NaN, devi capire **dove** nel grafo è successo.
> Servono strumenti di monitoring, profiling e testing matematico.

---

## 1. Custom Logger (Tensor Health Monitor)

### Obiettivo

Monitorare la "salute" dei tensori in tempo reale durante forward e backward: se qualcosa va storto, lo vedi subito.

### Metriche tracciate per ogni tensore

| Metrica | Formula | Allarme se |
|---|---|---|
| `mean` | $\mu = \frac{1}{n}\sum x_i$ | abs(mean) > 1e6 |
| `std` | $\sigma = \sqrt{\frac{1}{n}\sum(x_i-\mu)^2}$ | std > 1e6 oppure std = 0 |
| `min` | $\min(x)$ | -inf |
| `max` | $\max(x)$ | +inf |
| `has_nan` | `isnan(x_i)` | qualsiasi NaN presente |
| `has_inf` | `isinf(x_i)` | qualsiasi Inf presente |
| `sparsity` | % di zeri | > 99% (segnale morto) |

### Implementazione

```cpp
struct TensorStats {
    float mean, std, min, max;
    bool has_nan, has_inf;
    float sparsity;
    string name;
    int step;
};

class TensorLogger {
    enum Level { SILENT, SUMMARY, FULL, PANIC };
    Level level_ = SUMMARY;
    ofstream log_file_;
    vector<TensorStats> history_;

    TensorStats compute(const Tensor& t, const string& name, int step) {
        TensorStats s;
        s.name = name;
        s.step = step;
        float sum = 0, sum_sq = 0;
        s.min = FLT_MAX; s.max = -FLT_MAX;
        s.has_nan = false; s.has_inf = false;
        int zeros = 0;

        for (size_t i = 0; i < t.size(); i++) {
            float v = t.data()[i];
            if (isnan(v)) s.has_nan = true;
            if (isinf(v)) s.has_inf = true;
            sum += v;
            sum_sq += v * v;
            s.min = fmin(s.min, v);
            s.max = fmax(s.max, v);
            if (v == 0.0f) zeros++;
        }
        s.mean = sum / t.size();
        s.std = sqrtf(sum_sq / t.size() - s.mean * s.mean);
        s.sparsity = (float)zeros / t.size();
        return s;
    }

    void log(const Tensor& t, const string& name, int step) {
        auto s = compute(t, name, step);
        history_.push_back(s);

        if (s.has_nan || s.has_inf) {
            // PANIC: stampa stack trace, dumpa il tensore
            panic(s);
        } else if (level_ >= SUMMARY) {
            log_file_ << format("[step {}] {} | mean={:.4f} std={:.4f} "
                                "min={:.4f} max={:.4f} sparsity={:.1f}%\n",
                                step, name, s.mean, s.std,
                                s.min, s.max, s.sparsity * 100);
        }
    }
};
```

### Punti di inserzione

```cpp
// Nel forward di ogni layer
Tensor output = layer.forward(input);
logger.log(output, "layer_3.attention.output", global_step);

// Nei gradienti
logger.log(param.grad, "layer_3.ffn.W1.grad", global_step);
```

---

## 2. Profiling dell'Hardware

### NVIDIA Nsight Systems (GPU)

Analizza il timeline di esecuzione CPU/GPU per trovare colli di bottiglia.

```bash
# Profila l'esecuzione completa
nsys profile --trace=cuda,osrt --output=report ./my_llm --steps 10

# Visualizza il report
nsys-ui report.nsys-rep
```

| Cosa cercare | Problema |
|---|---|
| CPU gaps tra kernel launches | CPU troppo lenta a preparare dati |
| GPU idle tra kernel | Sync non necessari, kernel troppo piccoli |
| Long memcpy | Troppi trasferimenti CPU-GPU |
| Low occupancy | Kernel non sfrutta tutti i thread GPU |

### NVIDIA Nsight Compute (kernel-level)

```bash
# Profila un singolo kernel in dettaglio
ncu --set full -o kernel_report ./my_llm --steps 1
```

### Valgrind (CPU, memory)

```bash
# Memory leak detection
valgrind --tool=memcheck --leak-check=full ./my_llm

# Cache profiling
valgrind --tool=cachegrind ./my_llm
cg_annotate cachegrind.out.*
```

### perf (CPU, Linux)

```bash
# Hotspot profiling
perf record -g ./my_llm --steps 100
perf report

# Cache miss analysis
perf stat -e cache-misses,cache-references ./my_llm
```

### Profiler integrato nel codice

```cpp
class ScopedTimer {
    string name_;
    chrono::high_resolution_clock::time_point start_;
public:
    ScopedTimer(const string& name) : name_(name),
        start_(chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto end = chrono::high_resolution_clock::now();
        auto ms = chrono::duration_cast<chrono::microseconds>(end - start_);
        Profiler::instance().record(name_, ms.count());
    }
};

// Uso
void TransformerBlock::forward(Tensor& x) {
    { ScopedTimer t("attention"); attention(x); }
    { ScopedTimer t("ffn"); ffn(x); }
}

// Report
Profiler::instance().report();
// Output:
// attention:  45.2 ms (62%)
// ffn:        18.1 ms (25%)
// layernorm:   5.3 ms ( 7%)
// other:       4.4 ms ( 6%)
```

---

## 3. Unit Testing Matematico

### Confronto con NumPy

Genera reference values con Python/NumPy e confrontali con il C++:

```python
# generate_reference.py
import numpy as np
np.random.seed(42)
A = np.random.randn(64, 128).astype(np.float32)
B = np.random.randn(128, 64).astype(np.float32)
C = A @ B
# Salva in formato binario
A.tofile("test_data/matmul_A.bin")
B.tofile("test_data/matmul_B.bin")
C.tofile("test_data/matmul_C_ref.bin")
```

```cpp
// test_matmul_vs_numpy.cpp
TEST(BLAS, matmul_vs_numpy) {
    Tensor A = load_binary("test_data/matmul_A.bin", {64, 128});
    Tensor B = load_binary("test_data/matmul_B.bin", {128, 64});
    Tensor C_ref = load_binary("test_data/matmul_C_ref.bin", {64, 64});

    Tensor C = blas::matmul(A, B);

    // Confronto con tolleranza
    float max_err = 0;
    for (size_t i = 0; i < C.size(); i++)
        max_err = fmax(max_err, fabsf(C[i] - C_ref[i]));

    EXPECT_LT(max_err, 1e-4f)
        << "Max absolute error: " << max_err;
}
```

### Operazioni da testare vs NumPy

| Operazione C++ | Equivalente NumPy | Tolleranza |
|---|---|---|
| `matmul(A, B)` | `A @ B` | 1e-4 |
| `softmax(x)` | `scipy.special.softmax(x)` | 1e-5 |
| `layernorm(x)` | manuale con `np.mean`, `np.var` | 1e-5 |
| `gelu(x)` | `0.5*x*(1+np.tanh(...))` | 1e-6 |
| `attention(Q,K,V)` | manuale NumPy | 1e-3 |

---

## 4. NaN Debugger

Quando viene rilevato un NaN, stampa informazioni diagnostiche dettagliate:

```cpp
void panic(const TensorStats& s) {
    cerr << "=== NaN/Inf DETECTED ===" << endl;
    cerr << "Tensor: " << s.name << endl;
    cerr << "Step: " << s.step << endl;
    cerr << "Stats: mean=" << s.mean << " std=" << s.std << endl;
    cerr << "Range: [" << s.min << ", " << s.max << "]" << endl;

    // Dumpa il tensore per analisi offline
    dump_tensor_to_file(s.name + "_step" + to_string(s.step) + ".bin");

    // Stampa il grafo computazionale (path al NaN)
    print_computation_graph_path(s.name);

    // Opzionalmente: breakpoint per debugger
    #ifdef DEBUG
    __debugbreak();  // MSVC
    #endif
}
```

---

## 5. Struttura File

| File | Contenuto |
|---|---|
| `src/debug/tensor_logger.h/.cpp` | TensorLogger con salute e allarmi |
| `src/debug/profiler.h/.cpp` | ScopedTimer, report dei tempi |
| `src/debug/nan_debugger.h/.cpp` | NaN detection, panic, dump |
| `scripts/generate_reference.py` | Genera dati di riferimento NumPy |
| `test_data/` | File binari di riferimento |
| `tests/test_vs_numpy.cpp` | Confronto C++ vs NumPy |
| `tests/test_logger.cpp` | Test del logging system |

---

## 6. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_logger_nan` | NaN iniettato → logger entra in PANIC |
| `test_logger_stats` | Stats calcolate correttamente su tensore noto |
| `test_profiler` | ScopedTimer misura tempo ±10% |
| `test_matmul_vs_numpy` | Errore max < 1e-4 su matrici 64×128 |
| `test_softmax_vs_numpy` | Errore max < 1e-5 |
| `test_attention_vs_numpy` | Errore max < 1e-3 |
