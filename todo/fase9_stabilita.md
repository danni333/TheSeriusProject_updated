# Fase 9 — Stabilità Numerica (BF16/FP8 Mixed-Precision, Loss Scaling, µP Init)

> Senza i "paracadute" dei framework moderni, il modello può
> smettere di imparare all'improvviso: **Exploding Gradients** (numeri troppo grandi)
> o **Vanishing Gradients** (numeri troppo piccoli).

---

## 1. Gradient Clipping

### Problema

Durante la backpropagation, i gradienti possono crescere esponenzialmente layer dopo layer, causando aggiornamenti dei pesi enormi e divergenza del training.

### Soluzione: Max Norm Clipping

```
1. Calcola la norma L2 globale di tutti i gradienti:
   total_norm = √(Σᵢ ||gradᵢ||²)

2. Se total_norm > max_norm:
   scale = max_norm / total_norm
   gradᵢ *= scale   (per ogni parametro i)
```

### Implementazione

```cpp
float clip_grad_norm(vector<Tensor*>& params, float max_norm) {
    // Passo 1: norma L2 globale
    float total_norm_sq = 0.0f;
    for (auto* p : params) {
        if (!p->grad) continue;
        float* g = p->grad->data();
        size_t n = p->grad->size();
        // SIMD-accelerato
        for (size_t i = 0; i < n; i++)
            total_norm_sq += g[i] * g[i];
    }
    float total_norm = sqrtf(total_norm_sq);

    // Passo 2: clip
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto* p : params)
            if (p->grad)
                blas::scale(p->grad, scale);  // SIMD
    }
    return total_norm;  // utile per logging
}
```

| Iperparametro | Tipico | Note |
|---|---|---|
| `max_norm` | 1.0 | Standard per Transformer |

---

## 2. Weight Initialization

### Perché è importante

Pesi inizializzati male causano:

- **Troppo grandi** → attivazioni esplodono → loss = NaN
- **Troppo piccoli** → segnale muore nei primi layer → non impara

### Xavier (Glorot) Initialization

Per layer con attivazione lineare o tanh.

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}},\ +\sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

### He (Kaiming) Initialization

Per layer con ReLU/GELU.

$$W \sim \mathcal{N}\left(0,\ \sqrt{\frac{2}{n_{in}}}\right)$$

### Implementazione

```cpp
void xavier_uniform(Tensor& w, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    // Distribuzione uniforme [-limit, +limit]
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (size_t i = 0; i < w.size(); i++)
        w.data()[i] = dist(gen);
}

void he_normal(Tensor& w, int fan_in) {
    float std = sqrtf(2.0f / fan_in);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, std);
    for (size_t i = 0; i < w.size(); i++)
        w.data()[i] = dist(gen);
}
```

### Quale usare dove

| Layer | Inizializzazione |
|---|---|
| Embedding | `N(0, 0.02)` |
| Linear (pre-GELU) | He Normal |
| Linear (output projection) | Xavier Uniform |
| LayerNorm γ | `1.0` |
| LayerNorm β | `0.0` |
| Bias | `0.0` |

---

## 3. Loss Scaling (per Mixed Precision)

### Problema

In FP16, gradienti troppo piccoli diventano **zero** (underflow). Il modello smette di imparare.

### Soluzione: Dynamic Loss Scaling

```
scale = 65536.0  (iniziale, alto)

Per ogni step:
    loss_scaled = loss * scale
    backward(loss_scaled)           → gradienti amplificati
    gradienti /= scale              → torna alla scala corretta

    if (gradienti contengono NaN o Inf):
        scale /= 2                  → riduci la scala
        skip questo step
    else:
        optimizer.step()
        ogni N step senza overflow: scale *= 2
```

---

## 4. Stochastic Weight Averaging (SWA)

### Concetto

Mantieni una **media mobile esponenziale** (EMA) dei pesi durante il training. Il modello EMA è più robusto e generalizza meglio.

### Algoritmo

```
Per ogni step t:
    θ_ema = decay · θ_ema + (1 - decay) · θ_current

Tipicamente: decay = 0.999 o 0.9999
```

### Implementazione

```cpp
class EMAWeights {
    vector<Tensor> shadow_params;  // copie EMA dei pesi
    float decay;

    void update(const vector<Tensor*>& model_params) {
        for (size_t i = 0; i < shadow_params.size(); i++) {
            float* s = shadow_params[i].data();
            float* p = model_params[i]->data();
            size_t n = shadow_params[i].size();
            for (size_t j = 0; j < n; j++)
                s[j] = decay * s[j] + (1.0f - decay) * p[j];
        }
    }

    void apply_to(vector<Tensor*>& model_params) {
        // Copia shadow → model (per evaluation/inference)
    }
};
```

> [!NOTE]
> SWA viene usato solo per l'**inference finale**.
> Durante il training, si aggiornano i pesi normali.
> Il modello EMA viene usato per la valutazione e il checkpoint finale.

---

## 5. Struttura File

| File | Contenuto |
|---|---|
| `src/init/weight_init.h/.cpp` | Xavier, He, distribuzione custom |
| `src/stability/grad_clip.h/.cpp` | Gradient clipping (L2 norm) |
| `src/stability/loss_scaler.h/.cpp` | Dynamic Loss Scaling per FP16 |
| `src/stability/ema.h/.cpp` | Stochastic Weight Averaging / EMA |
| `tests/test_init.cpp` | Verifica range e distribuzione dei pesi |
| `tests/test_stability.cpp` | Gradient clipping, loss scaling, EMA |

---

## 6. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_xavier_range` | Pesi entro `±√(6/(fan_in+fan_out))` |
| `test_he_std` | Std dei pesi ≈ `√(2/fan_in)` |
| `test_clip_norm` | Dopo clipping, `||grad|| ≤ max_norm` |
| `test_clip_no_change` | Se `||grad|| < max_norm`, nessuna modifica |
| `test_loss_scale_overflow` | Gradiente NaN → scale dimezzata, step skippato |
| `test_loss_scale_growth` | N step senza overflow → scale raddoppiata |
| `test_ema_convergence` | EMA converge verso i pesi correnti |
| `test_ema_vs_original` | Modello EMA ha loss simile o migliore su validation |
| `test_training_stability` | 100 step di training: nessun NaN, loss decresce |
