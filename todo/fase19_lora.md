# Fase 19 — LoRA, QLoRA & DoRA (Parameter-Efficient Fine-Tuning)

> Invece di rifare il training da zero, LoRA aggiunge piccole matrici
> "adattatrici" che modificano il comportamento del modello con costi minimi.

---

## 1. LoRA (Low-Rank Adaptation)

### Concetto

Invece di aggiornare la matrice dei pesi $W$ (enorme), la decomponi:

$$W' = W + \Delta W = W + B \cdot A$$

dove:

- $W$: pesi originali frozen (non si toccano) — shape $(d_{out}, d_{in})$
- $A$: matrice down-projection — shape $(r, d_{in})$, con $r \ll d$
- $B$: matrice up-projection — shape $(d_{out}, r)$
- $r$: **rank** (tipicamente 8-64), controlla la capacità

```
             d_in
         ┌───────────┐
         │            │
d_out    │   W (frozen)│ +
         │            │
         └───────────┘
                         d_in          d_in
                     ┌──────┐      ┌──────┐
                     │  A   │  ×   │      │
                r    │(r×d_in)│     │  x   │
                     └──┬───┘      └──────┘
                        │
                     ┌──▼───┐
                d_out│  B   │ = ΔW·x
                     │(d_out×r)│
                     └──────┘
```

### Parametri da trainare

| | Full fine-tuning | LoRA (rank=16) |
|---|---|---|
| Modello 7B | 7B parametri | ~17M parametri |
| VRAM | ~56 GB | ~1 GB (solo A, B + optimizer) |

### Implementazione

```cpp
class LoRALinear {
    Tensor& W;          // pesi originali (frozen)
    Tensor A, B;        // adattatori (trainable)
    float alpha;        // scaling factor
    int rank;

    Tensor forward(const Tensor& x) {
        // Output = W·x + (alpha/rank) · B·(A·x)
        Tensor base = matmul(W, x);        // frozen
        Tensor lora = matmul(A, x);        // (rank, seq)
        lora = matmul(B, lora);             // (d_out, seq)
        return base + (alpha / rank) * lora;
    }
};
```

### Inizializzazione

- $A$: inizializzato con distribuzione gaussiana $\mathcal{N}(0, \sigma^2)$
- $B$: inizializzato a **zero** → all'inizio $\Delta W = 0$, il modello parte identico all'originale

### Dove applicare LoRA

| Layer | Applicare LoRA? |
|---|---|
| `q_proj`, `v_proj` | ✅ Sì (standard) |
| `k_proj`, `o_proj` | ✅ Opzionale |
| `gate_proj`, `up_proj`, `down_proj` | ✅ Per risultati migliori |
| Embedding, LayerNorm | ❌ No |

### Merge dei pesi (per inference)

```cpp
void merge_lora(Tensor& W, const LoRALinear& lora) {
    // W = W + (alpha/rank) * B · A
    Tensor delta = matmul(lora.B, lora.A);
    W += (lora.alpha / lora.rank) * delta;
}
// Dopo il merge, inference = velocità originale (no overhead)
```

---

## 2. Online Learning

### Concetto

Il modello impara dalle **correzioni dell'utente** salvando piccoli checkpoint LoRA locali.

### Flusso

```
Utente: "Qual è la capitale della Francia?"
Modello: "Lione"  ← sbagliato
Utente: [Corregge] "La risposta corretta è Parigi"
    │
    ▼
Sistema:
  1. Crea training sample: (domanda, "Parigi")
  2. Forward + backward solo sui pesi LoRA
  3. Aggiorna A, B (pochi step, lr basso)
  4. Salva checkpoint incrementale: lora_patch_001.bin
```

### Implementazione

```cpp
class OnlineLearner {
    Model& model;
    LoRAAdapter& lora;
    AdamW optimizer;  // solo per parametri LoRA
    int patch_counter = 0;

    void learn_correction(const string& input, const string& correct) {
        auto ids = tokenizer.encode(format_correction(input, correct));
        
        // Forward
        Tensor loss = model.forward_loss(ids);
        
        // Backward (solo LoRA params hanno requires_grad)
        loss.backward();
        
        // Update (pochi step)
        for (int i = 0; i < 3; i++) {
            optimizer.step();
            optimizer.zero_grad();
        }
        
        // Salva patch incrementale
        save_lora_patch(format("lora_patch_{:04d}.bin", ++patch_counter));
    }
};
```

### Stack di LoRA adapter

```
Base model (frozen)
  + LoRA "italiano" (lingua)
  + LoRA "medicina" (dominio)
  + LoRA "user_corrections" (online learning)
  = Modello personalizzato
```

```cpp
Tensor forward(const Tensor& x) {
    Tensor out = matmul(W, x);
    for (auto& adapter : active_adapters)
        out += adapter.forward(x);
    return out;
}
```

---

## 3. Multi-LoRA Serving (Più Adapter in Contemporanea)

> In produzione servono più "personalità" dello stesso modello:
> un LoRA per italiano, uno per codice, uno per medicina.
> Multi-LoRA carica un unico modello base e switcha adapter per-request.

### Architettura

```text
          Modello Base (7B, in VRAM una sola volta)
          ┌────────────────────────────────────┐
          │                                    │
          │  ┌─────────┐ ┌─────────┐ ┌──────┐ │
          │  │ LoRA IT │ │ LoRA EN │ │LoRA  │ │
          │  │ (17 MB) │ │ (17 MB) │ │Code  │ │
          │  └─────────┘ └─────────┘ │(17MB)│ │
          │                          └──────┘ │
          └────────────────────────────────────┘

Request A: "Traduci in italiano"  → usa LoRA IT
Request B: "Write a Python func"  → usa LoRA Code
Request C: "Explain relativity"   → usa LoRA EN
→ Tutti e 3 serviti in parallelo con lo STESSO modello base
```

### Implementazione

```cpp
namespace serius::lora {

class MultiLoRAServer {
    Model& base_model_;  // caricato una sola volta
    map<string, LoRAAdapter> adapters_;  // nome → adapter

public:
    void load_adapter(const string& name, const string& path) {
        adapters_[name] = LoRAAdapter::load(path);
        spdlog::info("LoRA '{}' loaded ({} MB)", name,
                     adapters_[name].size_bytes() / 1024 / 1024);
    }

    void unload_adapter(const string& name) {
        adapters_.erase(name);
    }

    // Inference con adapter specifico per-request
    string generate(const string& prompt, const string& adapter_name,
                    const SamplerConfig& config) {
        LoRAAdapter* lora = nullptr;
        if (!adapter_name.empty()) {
            auto it = adapters_.find(adapter_name);
            if (it != adapters_.end())
                lora = &it->second;
        }
        return base_model_.generate(prompt, config, lora);
    }

    // Hot-swap: cambia adapter senza riavviare il server
    void hot_swap(const string& name, const string& new_path) {
        adapters_[name] = LoRAAdapter::load(new_path);
    }
};

} // namespace serius::lora
```

### API Endpoint

```json
// POST /v1/chat/completions
{
    "model": "default",
    "messages": [...],
    "lora_adapter": "italiano"   // ← seleziona l'adapter
}
```

### File aggiuntivi

| File | Contenuto |
| --- | --- |
| `src/lora/multi_lora_server.h/.cpp` | Multi-LoRA serving con routing |
| `src/lora/hot_swap.h/.cpp` | Hot-swap adapter a runtime |
| `tests/test_multi_lora.cpp` | 3 adapter in parallelo, output diverso |

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `src/lora/lora_linear.h/.cpp` | LoRALinear: forward, merge, init |
| `src/lora/lora_adapter.h/.cpp` | Adapter manager (applica LoRA a layer selezionati) |
| `src/lora/lora_io.h/.cpp` | Save/load checkpoint LoRA |
| `src/learning/online.h/.cpp` | OnlineLearner: correzioni utente |
| `tests/test_lora.cpp` | Rank, merge, multi-adapter |
| `tests/test_online.cpp` | Correzione → modello impara |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_lora_init` | B=0 all'inizio → output = base model |
| `test_lora_rank` | ΔW ha rank ≤ r |
| `test_lora_merge` | Dopo merge: output identico a LoRA forward |
| `test_lora_finetune` | 50 step su toy data: loss decresce |
| `test_multi_adapter` | 2 adapter stacked: output = base + Δ₁ + Δ₂ |
| `test_online_learn` | Dopo correzione, modello risponde correttamente |
| `test_lora_save_load` | Save → load: pesi identici |
