# Fase 5 — Ottimizzazione e Training Loop

> Il modello "impara" aggiornando iterativamente i pesi
> per ridurre la loss. Serve un optimizer, un sistema di
> salvataggio e, opzionalmente, parallelismo multi-core.

---

## 1. Optimizer: AdamW

### Perché AdamW (e Muon come alternativa SOTA)

SGD semplice è lento a convergere per modelli grandi. **AdamW** mantiene medie mobili dei gradienti (momento) e dei gradienti al quadrato (velocità adattiva), con **weight decay decoupled**.

> [!TIP]
> **Muon Optimizer** (2025): Basato sull'ortogonalizzazione delle matrici di aggiornamento tramite Newton-Schulz iteration, Muon raggiunge circa **2× l'efficienza computazionale** di AdamW nei regimi compute-optimal. Utilizzato per addestrare Moonlight (3B/16B MoE). Meno sensibile agli iperparametri di AdamW.

### Algoritmo (per ogni parametro θ)

```
Ad ogni step t:
    g = ∂L/∂θ                              // gradiente

    m = β₁ · m + (1 - β₁) · g              // media mobile del gradiente
    v = β₂ · v + (1 - β₂) · g²             // media mobile del gradiente²

    m̂ = m / (1 - β₁ᵗ)                      // bias correction
    v̂ = v / (1 - β₂ᵗ)                      // bias correction

    θ = θ - lr · (m̂ / (√v̂ + ε) + λ · θ)   // update con weight decay (λ)
```

### Iperparametri

| Parametro | Default | Descrizione |
|---|---|---|
| `lr` | 3e-4 | Learning rate |
| `β₁` | 0.9 | Decadimento media mobile 1° momento |
| `β₂` | 0.999 | Decadimento media mobile 2° momento |
| `ε` | 1e-8 | Stabilità numerica |
| `λ` (weight_decay) | 0.01 | Forza del weight decay |

### Stato dell'optimizer

Per ogni parametro, AdamW mantiene:

```cpp
struct ParamState {
    Tensor m;    // 1° momento (stesso shape del parametro)
    Tensor v;    // 2° momento (stesso shape del parametro)
    int step;    // contatore degli step
};
```

> [!WARNING]
> Lo stato dell'optimizer **raddoppia** la memoria necessaria:
> per ogni tensore di pesi, servono `m` e `v` della stessa dimensione.

---

### Warmup-Stable-Decay (WSD) — Lo standard di frontiera

A differenza del cosine decay classico, il **WSD scheduler** (usato da DeepSeek, Moonlight) **non richiede di fissare in anticipo il numero totale di step**, permettendo training continuo e branching da checkpoint.

```text
Fase 1 — Warmup:  lr ramp-up lineare (come prima)
Fase 2 — Stable:  lr COSTANTE al valore massimo (la fase più lunga)
Fase 3 — Decay:   breve decadimento finale solo quando si vuole convergere un checkpoint
```

```text
lr
 │  /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
 │ /                       \_____
 │/                              
 └────────────────────────────────▶ step
   warmup   stable (lungo)  decay
```

> [!TIP]
> **Vantaggio chiave**: si può lanciare il training senza `max_steps`, e poi decidere in un secondo momento quando applicare il decay finale. Permette anche di "forkare" checkpoint multipli a diversi livelli di convergenza.

---

## 3. Checkpointing

### Salvataggio (serializzazione binaria)

```
Formato file .ckpt:
┌────────────────────────┐
│ Magic number (4 bytes) │  "TSCP"
│ Version     (4 bytes)  │
│ Num params  (4 bytes)  │
├────────────────────────┤
│ Param 0: name_len      │
│ Param 0: name (UTF-8)  │
│ Param 0: ndim          │
│ Param 0: shape[]       │
│ Param 0: data (float[])│  ← dati raw, no overhead
├────────────────────────┤
│ Param 1: ...           │
└────────────────────────┘
```

### Funzioni

```cpp
void save_checkpoint(const string& path,
                     const Model& model,
                     const AdamW& optimizer,
                     int epoch, int step);

void load_checkpoint(const string& path,
                     Model& model,
                     AdamW& optimizer,
                     int& epoch, int& step);
```

> [!TIP]
> Salva anche lo stato dell'optimizer (m, v, step).
> Senza di esso, riprendere il training ricomincerebbe
> il warmup delle medie mobili, degradando la convergenza.

---

## 4. Parallelismo CPU con OpenMP

### Dove applicarlo

| Punto | Direttiva |
|---|---|
| Matmul (loop esterno) | `#pragma omp parallel for` |
| Elementwise ops | `#pragma omp parallel for simd` |
| DataLoader (lettura file) | Thread pool dedicato |
| Gradient accumulation | `#pragma omp parallel for reduction(+:...)` |

### Esempio

```cpp
// Matmul parallelizzata
#pragma omp parallel for collapse(2)
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A(i, k) * B(k, j);
        C(i, j) = sum;
    }
}
```

### Flag CMake

```cmake
find_package(OpenMP REQUIRED)
target_link_libraries(my_target PUBLIC OpenMP::OpenMP_CXX)
```

---

## 5. Training Loop Completo

```
Carica dati (DataLoader)
Inizializza modello (random weights)
Inizializza optimizer (AdamW)

Per ogni epoch:
    Per ogni batch:
        1. Forward pass       → loss
        2. Backward pass      → gradienti
        3. Gradient clipping  → max_norm = 1.0
        4. Optimizer step     → aggiorna pesi
        5. Zero grad          → resetta gradienti
        6. Log (loss, lr, step)

    Salva checkpoint ogni N epoch
    Valutazione su validation set (opzionale)
```

### Gradient Clipping

```cpp
void clip_grad_norm(vector<Tensor*>& params, float max_norm) {
    float total_norm = 0;
    for (auto* p : params)
        total_norm += sum(p->grad * p->grad);
    total_norm = sqrt(total_norm);
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (auto* p : params)
            p->grad *= scale;
    }
}
```

---

## 6. Struttura File

| File | Contenuto |
|---|---|
| `src/optim/adamw.h/.cpp` | Optimizer AdamW |
| `src/optim/lr_schedule.h/.cpp` | Learning rate scheduler |
| `src/training/trainer.h/.cpp` | Training loop, gradient clipping |
| `src/training/checkpoint.h/.cpp` | Save/load checkpoint binario |
| `tests/test_adamw.cpp` | Convergenza su toy problem |
| `tests/test_checkpoint.cpp` | Save → load → verifica uguaglianza |

---

## 7. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_adamw_step` | Un singolo step riduce la loss su un toy problem |
| `test_adamw_convergence` | 100 step su `f(x) = x²` → `x ≈ 0` |
| `test_weight_decay` | I pesi decadono verso zero anche senza gradiente |
| `test_lr_schedule` | Il learning rate segue la curva warmup + cosine |
| `test_checkpoint_roundtrip` | `save → load → save`: i due file sono identici |
| `test_gradient_clipping` | La norma del gradiente non supera `max_norm` |
| `test_training_loop` | 10 step di training su dati sintetici: loss decresce |
