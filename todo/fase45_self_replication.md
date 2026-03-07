# Fase 45 — Self-Replication & Recursive Code Improvement

> Il sistema gestisce il proprio codice: rileva bottleneck,
> scrive versioni ottimizzate, testa, e si aggiorna live.

---

## 1. Auto-Refactoring Pipeline

```
1. Profiler rileva bottleneck (Fase 12)
     │
     ▼
2. Modello analizza il kernel lento
     │
     ▼
3. Genera versione v2 ottimizzata
     │
     ▼
4. Compila in sandbox
     │
     ▼
5. Esegue test di correttezza (vs. output reference)
     │
     ▼
6. Benchmark: v2 più veloce di v1?
     │
     ├── Sì → Hot-swap del binario
     └── No → Scarta e riprova
```

### Implementazione

```cpp
class AutoRefactorer {
    Model& coding_model;
    Profiler& profiler;

    bool optimize_hotspot(const string& kernel_source, const string& kernel_name) {
        // 1. Analisi performance
        auto perf = profiler.get_stats(kernel_name);

        // 2. Genera codice ottimizzato
        string prompt = format(
            "Kernel: {}\nLatency: {}ms\nBottleneck: {}\n"
            "Genera una versione ottimizzata:\n{}",
            kernel_name, perf.avg_ms, perf.bottleneck, kernel_source);
        string new_code = coding_model.generate(prompt, {.temp=0.2});

        // 3. Compila in sandbox
        if (!compile_in_sandbox(new_code, kernel_name + "_v2"))
            return false;

        // 4. Test correttezza
        if (!verify_correctness(kernel_name, kernel_name + "_v2"))
            return false;

        // 5. Benchmark
        float old_ms = benchmark(kernel_name);
        float new_ms = benchmark(kernel_name + "_v2");

        if (new_ms < old_ms * 0.95f) { // almeno 5% più veloce
            hot_swap(kernel_name, kernel_name + "_v2");
            git_commit(kernel_name, old_ms, new_ms);
            return true;
        }
        return false;
    }
};
```

---

## 2. Gestione Git autonoma

```cpp
class GitManager {
    string repo_path;

    void commit_optimization(const string& kernel,
                              float old_ms, float new_ms) {
        string msg = format("perf: {} {:.1f}ms -> {:.1f}ms ({:.0f}% faster)",
                            kernel, old_ms, new_ms,
                            (1 - new_ms/old_ms) * 100);
        system(format("cd {} && git add -A && git commit -m '{}'",
                       repo_path, msg).c_str());
    }
};
```

---

## 3. Hot-Swapping

```cpp
// Carica libreria dinamica senza riavviare
void hot_swap(const string& old_kernel, const string& new_kernel) {
    #ifdef _WIN32
    void* lib = LoadLibraryA((new_kernel + ".dll").c_str());
    auto fn = (KernelFunc)GetProcAddress(lib, "kernel_entry");
    #else
    void* lib = dlopen((new_kernel + ".so").c_str(), RTLD_NOW);
    auto fn = (KernelFunc)dlsym(lib, "kernel_entry");
    #endif
    kernel_registry[old_kernel] = fn;
    // Il vecchio kernel viene scaricato al prossimo ciclo GC
}
```

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `src/autoopt/refactorer.h/.cpp` | Pipeline auto-refactoring |
| `src/autoopt/sandbox.h/.cpp` | Compilazione e test in sandbox |
| `src/autoopt/hot_swap.h/.cpp` | Dynamic library loading |
| `src/autoopt/git_manager.h/.cpp` | Commit automatici |
| `tests/test_autoopt.cpp` | Refactoring produce codice corretto e veloce |

---

## 5. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_sandbox_compile` | Codice generato compila senza errori |
| `test_correctness` | Output v2 = output v1 (atol 1e-5) |
| `test_speedup` | v2 non più lenta di v1 |
| `test_hot_swap` | Kernel sostituito a runtime senza crash |
| `test_rollback` | Se v2 fallisce, v1 resta attiva |

