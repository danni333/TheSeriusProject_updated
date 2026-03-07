# Fase 38 — Self-Evolution, NAS & SSM/Mamba-2 Hybrid Auto-Optimization

> Superare il limite quadratico del Transformer.
> SSM/Mamba per memoria lineare, Auto-Optimization per AI che ottimizza AI.

---

## 1. State Space Models (SSM) / Mamba

### Problema del Transformer

Attention: $O(n^2)$ in memoria — impossibile per milioni di token.

### SSM: memoria lineare $O(n)$

```text
Input x1, x2, ..., x?

Stato nascosto h evolve linearmente:
  h? = A · h??1 + B · x?        (transizione di stato)
  y? = C · h? + D · x?           (output)

A, B, C, D: matrici parametriche apprese
```

### Mamba: SSM selettivo

```cpp
struct MambaBlock {
    // Parametri input-dipendenti (selective)
    Tensor compute_A(const Tensor& x);  // discretizzazione
    Tensor compute_B(const Tensor& x);
    Tensor compute_C(const Tensor& x);
    float dt;  // step size

    Tensor forward(const Tensor& x) {
        int seq = x.shape()[0], d = x.shape()[1];
        Tensor h = Tensor::zeros({d});  // stato nascosto
        Tensor output({seq, d});

        for (int t = 0; t < seq; t++) {
            Tensor A = compute_A(x[t]);
            Tensor B = compute_B(x[t]);
            Tensor C = compute_C(x[t]);
            // Discretizzazione con zero-order hold
            Tensor A_bar = exp(A * dt);
            Tensor B_bar = (A_bar - 1) * inv(A) * B;
            // Update stato
            h = A_bar * h + B_bar * x[t];
            output[t] = C * h;
        }
        return output;
    }
};
```

### Vantaggi

| | Transformer | Mamba/Mamba-2 |
| --- | --- | --- |
| Memoria | $O(n^2)$ | $O(n)$ |
| Inference | Serve KV Cache | Solo stato h |
| 1M token | Impossibile | Fattibile |

### Mamba-2: Structured State Space Duality (SSD)

L'innovazione chiave di **Mamba-2** (2024, Dao & Gu) è la **Structured State Space Duality**: una equivalenza matematica precisa tra SSM strutturati e una forma di masked attention.

```text
SSM selettivo   ⟺   Masked Attention (semi-separabile)

Se A è scalare-identità: il calcolo SSM può essere espresso
come una matmul su tensor core (GEMM), sfruttando WGMMA su Hopper.
```

**Vantaggi di Mamba-2 vs Mamba-1:**

- **2-8× più veloce** nel training (tensor core utilization)
- **State size N = 64-256** (vs N=16 in Mamba-1) senza sacrificare la velocità
- **Multi-head SSM blocks**: analogia con multi-head attention, ogni head ha il proprio stato
- Supporto nativo per **tensor parallelism** e **sequence parallelism**

```cpp
struct Mamba2Block {
    int n_heads;          // multi-head SSM (come multi-head attention)
    int head_dim;         // P=64 default (vs P=1 in Mamba-1)
    int state_size;       // N=64 o 128 (vs N=16 in Mamba-1)

    // A è scalare×identità per tutte le dimensioni dello stato
    // → permette dual-form computation via matmul
    Tensor A_log;  // log(A), appreso come parametro

    Tensor forward(const Tensor& x) {
        // Proietta input → (B, C, dt) come in Mamba-1, ma PARALLELO
        auto [B, C, dt] = project_params(x);
        Tensor A = -exp(A_log);  // scalare negativo per stabilità

        // Dual computation: può usare RECURRENCE o ATTENTION form
        if (training) {
            // Attention form: sfrutta tensor cores con matmul
            return ssd_attention_form(x, A, B, C, dt);
        } else {
            // Recurrence form: O(1) per step durante inference
            return ssd_recurrence_form(x, A, B, C, dt);
        }
    }
};
```

### Jamba: Architettura Ibrida SSM + Transformer

Il pattern emergente nel 2024-2025 è l'architettura **ibrida** che alterna layer Mamba e layer Transformer (come Jamba, Zamba):

```text
Layer 1: Mamba-2 Block   ← memoria lineare
Layer 2: Mamba-2 Block
Layer 3: Transformer Block (Full Attention + MoE FFN)  ← cattura pattern globali
Layer 4: Mamba-2 Block
Layer 5: Mamba-2 Block
Layer 6: Transformer Block (Full Attention + MoE FFN)
...
Ratio tipico: ~6:1 (Mamba : Transformer)
```

**Vantaggi dell'ibrido:**

- Memoria combinata: O(n) dei layer Mamba + O(n²) solo nei pochi layer Transformer
- Qualità pari o superiore a Transformer puro su language modeling
- Throughput 3× superiore su sequenze lunghe (128k+ token)

---

## 2. Auto-Optimization (AI che ottimizza AI)

```cpp
class KernelOptimizer {
    Model& coding_model;  // modello che genera codice CUDA

    string optimize_kernel(const string& current_kernel,
                            const string& performance_data) {
        string prompt = format(
            "Questo kernel CUDA ha latenza {}ms. "
            "Ottimizzalo per ridurre gli accessi a HBM:\n{}\n"
            "Genera la versione ottimizzata:",
            performance_data, current_kernel);
        return coding_model.generate(prompt, {.temp=0.2, .max_tokens=2048});
    }

    bool validate_and_benchmark(const string& new_kernel) {
        // 1. Compila con nvcc
        // 2. Verifica correttezza (output = reference)
        // 3. Benchmark: se più veloce, adotta
        return new_latency < old_latency * 0.95f;  // 5% improvement
    }
};
```

---

## 3. Struttura File

| File | Contenuto |
| --- | --- |
| `src/ssm/mamba.h/.cpp` | Mamba block (SSM selettivo) |
| `src/ssm/ssm_scan.cu` | Parallel scan CUDA per SSM |
| `src/autoopt/kernel_optimizer.h/.cpp` | Auto-optimization loop |
| `tests/test_mamba.cpp` | Forward/backward, confronto vs attention |

---

## 4. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_mamba_linear` | Memoria cresce linearmente con seq_len |
| `test_mamba_quality` | Perplexity comparabile a Transformer su test set |
| `test_ssm_scan` | Parallel scan = sequential scan |
| `test_autoopt` | Kernel ottimizzato è più veloce e corretto |

---

> Invece di decidere manualmente quanti layer o teste usare,
> il sistema testa e sceglie la configurazione ottimale.

---

## 1. Auto-Tuning dei Kernel CUDA

### Concetto

Testa diverse configurazioni di blocchi/thread e scegli la più veloce per l'hardware specifico.

```cpp
struct KernelConfig {
    int block_x, block_y;
    int tile_size;
    bool use_shared_mem;
};

KernelConfig auto_tune_matmul(int M, int N, int K) {
    vector<KernelConfig> candidates = {
        {16, 16, 32, true},
        {32, 8,  64, true},
        {32, 32, 32, false},
        {16, 16, 64, true},
        {8,  8, 128, true},
    };
    KernelConfig best;
    float best_time = INFINITY;
    for (auto& cfg : candidates) {
        // Warmup
        launch_matmul(cfg, M, N, K);
        cudaDeviceSynchronize();
        // Benchmark
        auto t0 = now();
        for (int i = 0; i < 100; i++)
            launch_matmul(cfg, M, N, K);
        cudaDeviceSynchronize();
        float elapsed = since(t0) / 100.0f;
        if (elapsed < best_time) {
            best_time = elapsed;
            best = cfg;
        }
    }
    return best;
}
```

### Cache dei risultati

```cpp
// Salva configurazioni ottimali per riuso
map<tuple<int,int,int>, KernelConfig> tune_cache;
string cache_file = "tuning_cache_" + gpu_name() + ".json";
```

---

## 2. Pruning Dinamico

### Approccio

Identifica neuroni "morti" o inutili durante l'inference e li rimuove.

```cpp
class DynamicPruner {
    float threshold;

    // Structured pruning: rimuovi intere righe/colonne
    Tensor prune_linear(const Tensor& W, const Tensor& activation_stats) {
        // activation_stats: media |attivazione| per ogni neurone
        vector<int> keep;
        for (int i = 0; i < W.shape()[0]; i++)
            if (activation_stats[i] > threshold)
                keep.push_back(i);

        // Crea peso ridotto
        Tensor W_pruned({(int)keep.size(), W.shape()[1]});
        for (int i = 0; i < keep.size(); i++)
            W_pruned[i] = W[keep[i]];
        return W_pruned;
    }

    // Calcola statistiche di attivazione su calibration set
    Tensor collect_stats(Model& model, const vector<string>& calib_data) {
        Tensor stats = Tensor::zeros({d_ff});
        for (auto& text : calib_data) {
            auto acts = model.forward_with_hooks(tokenizer.encode(text));
            stats += abs_mean(acts);
        }
        return stats / calib_data.size();
    }
};
```

### Risultati tipici

| Pruning ratio | Speedup | Perdita accuracy |
| --- | --- | --- |
| 10% | 1.1x | ~0% |
| 30% | 1.4x | ~0.5% |
| 50% | 1.8x | ~2% |
| 70% | 2.5x | ~5% |

---

## 3b. Struttura File

| File | Contenuto |
| --- | --- |
| `src/automl/auto_tune.h/.cpp` | Auto-tuning kernel CUDA |
| `src/automl/pruner.h/.cpp` | Pruning dinamico (structured) |
| `src/automl/tune_cache.h/.cpp` | Cache risultati tuning per GPU |
| `tests/test_automl.cpp` | Tuning trova config valida, pruning mantiene accuracy |

---

## 4b. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_auto_tune` | Config scelta è la più veloce (±5%) |
| `test_tune_cache` | Risultati salvati e ricaricati correttamente |
| `test_prune_accuracy` | 30% pruning → accuracy cala < 1% |
| `test_prune_speedup` | 30% pruning → speedup > 1.3x |
