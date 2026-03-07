# Fase 30 — Kernel Fusion & JIT Graph Compilation (Triton-style)

> Il segreto di TensorRT e Xformers: fondere operazioni
> e pre-compilare il grafo per eliminare overhead di runtime.

---

## 1. Operator Fusion

### Problema

Ogni kernel CUDA ha overhead di lancio (~5-10μs). Con centinaia di operazioni per forward pass, il tempo si accumula. Inoltre, ogni kernel legge/scrive HBM separatamente.

```
Senza fusion:
  LayerNorm: HBM → SRAM → HBM     (lettura + scrittura)
  GELU:      HBM → SRAM → HBM     (lettura + scrittura)
  Dropout:   HBM → SRAM → HBM     (lettura + scrittura)
  = 6 accessi HBM

Con fusion:
  FusedNormGeluDropout: HBM → SRAM → HBM  (1 lettura + 1 scrittura)
  = 2 accessi HBM → 3x più veloce
```

### Kernel fusi comuni

```cpp
// Fused: bias + GELU
__global__ void fused_bias_gelu(float* x, const float* bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i] + bias[i % bias_size];
        // GELU approssimata
        x[i] = 0.5f * val * (1.0f + tanhf(0.7978846f * (val + 0.044715f * val*val*val)));
    }
}

// Fused: LayerNorm + proiezione lineare
__global__ void fused_layernorm_linear(
    const float* input, const float* gamma, const float* beta,
    const float* weight, float* output, int d_model, int d_out) {
    // Shared memory per calcolo media/varianza
    // ...poi matmul subito dopo, senza uscire dalla SRAM
}

// Fused: Softmax + Mask (per attention)
__global__ void fused_masked_softmax(float* scores, const bool* mask,
                                      int seq_len, float scale) {
    // Scala, applica mask e softmax in un unico kernel
}
```

---

## 2. Static Graph Compilation

### Concetto

Prima dell'inference, analizza l'architettura del Transformer e pre-calcola il percorso ottimale.

```cpp
class GraphCompiler {
    struct Op {
        string type;  // "matmul", "layernorm", "gelu"...
        vector<int> inputs, outputs;
        TensorShape output_shape;
    };

    vector<Op> ops;

    // Fase 1: Traccia il grafo
    void trace(Model& model, TensorShape input_shape) {
        // Esegue un forward pass "dummy" registrando tutte le operazioni
    }

    // Fase 2: Ottimizza il grafo
    void optimize() {
        fuse_elementwise_chains();    // bias+gelu+dropout → 1 kernel
        fuse_layernorm_linear();      // layernorm+matmul → 1 kernel
        eliminate_redundant_copies(); // rimuovi copie inutili
        reorder_for_locality();       // riordina per cache locality
        preallocate_buffers();        // alloca tutta la memoria upfront
    }

    // Fase 3: Compila in un piano di esecuzione
    ExecutionPlan compile() {
        ExecutionPlan plan;
        for (auto& op : ops)
            plan.add(select_best_kernel(op));
        return plan;
    }
};

// Esecuzione: zero overhead di gestione
class ExecutionPlan {
    vector<KernelLaunch> kernels;  // pre-configurati
    void* workspace;               // memoria pre-allocata

    void execute(const Tensor& input, Tensor& output) {
        for (auto& k : kernels)
            k.launch();  // nessuna decisione a runtime
    }
};
```

### Pattern di fusione riconosciuti

| Pattern | Fuso in |
|---|---|
| `bias + GELU` | `fused_bias_gelu` |
| `LayerNorm + Linear` | `fused_ln_linear` |
| `Softmax + Mask + Scale` | `fused_masked_softmax` |
| `Residual + LayerNorm` | `fused_residual_ln` |
| `QKV projection (3 matmul)` | `fused_qkv_proj` (1 matmul grande) |

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/optim/fused_kernels.cu` | Kernel CUDA fusi |
| `src/optim/graph_compiler.h/.cpp` | Tracciamento e ottimizzazione grafo |
| `src/optim/execution_plan.h/.cpp` | Piano di esecuzione pre-compilato |
| `tests/test_fusion.cpp` | Output fuso = output separato |
| `tests/test_compiler.cpp` | Grafo compilato produce output corretto |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_fused_bias_gelu` | Output = bias + gelu separati (atol 1e-5) |
| `test_fused_softmax` | Output = mask + scale + softmax separati |
| `test_graph_trace` | Grafo tracciato ha il numero corretto di op |
| `test_compiled_output` | Output compilato = output eager |

### Benchmark

| Benchmark | Misura |
|---|---|
| `bench_fused_vs_separate` | Speedup kernel fusi vs separati |
| `bench_compiled_vs_eager` | Latenza: grafo compilato vs esecuzione dinamica |
