# Fase 43 — World Models & V-JEPA (Latent Predictive Architecture)

> L'LLM non prevede solo la parola successiva: simula le conseguenze
> di un'azione nello spazio latente prima di rispondere.

---

## 1. Predictive World Models

### Architettura

```
Stato attuale s_t  +  Azione a_t
         │                │
         ▼                ▼
    ┌─────────────────────────┐
    │    World Model (WM)     │
    │    s_{t+1} = WM(s_t, a_t)│
    └────────────┬────────────┘
                 │
                 ▼
    Stato predetto s_{t+1}  →  Reward predetto r_{t+1}
```

### Implementazione

```cpp
class WorldModel {
    NeuralODE dynamics;    // evoluzione stato
    Decoder decoder;       // stato → predizione osservabile

    struct Prediction {
        Tensor next_state;
        float predicted_reward;
        float uncertainty;
    };

    Prediction predict(const Tensor& state, const Tensor& action) {
        Tensor combined = concat(state, action);
        Tensor next = dynamics.forward(combined, 0, 1, /*steps=*/10);
        float reward = reward_head(next);
        float unc = uncertainty_head(next);
        return {next, reward, unc};
    }

    // "Immaginazione": simula N step nel futuro
    vector<Prediction> imagine(const Tensor& state,
                                const vector<Tensor>& action_sequence) {
        vector<Prediction> trajectory;
        Tensor s = state;
        for (auto& a : action_sequence) {
            auto pred = predict(s, a);
            trajectory.push_back(pred);
            s = pred.next_state;
        }
        return trajectory;
    }
};
```

---

## 2. JEPA (Joint-Embedding Predictive Architecture)

### Differenza da modelli generativi

```
Generativo (GPT): predice i pixel/token esatti → costoso, fragile
JEPA (LeCun):     predice le RAPPRESENTAZIONI → robusto, efficiente

Input (video frame) → Encoder → z_t
                                 │
                    Predictor: z_{t+1} = f(z_t, azione)
                                 │
Next frame → Encoder → z_{t+1}_target
                                 │
Loss = ||z_{t+1}_pred - z_{t+1}_target||²
```

```cpp
class JEPA {
    Encoder encoder;        // immagine/testo → embedding
    Encoder target_encoder; // EMA dell'encoder (non trainato direttamente)
    Predictor predictor;    // predice embedding futuro

    float train_step(const Tensor& x_t, const Tensor& x_next) {
        // Encoder online
        Tensor z_t = encoder.forward(x_t);
        Tensor z_pred = predictor.forward(z_t);
        // Target encoder (EMA, no gradient)
        Tensor z_target = target_encoder.forward(x_next).detach();
        // Loss nello spazio latente
        return mse_loss(z_pred, z_target);
    }

    void update_target_encoder(float ema_decay = 0.999f) {
        // EMA update (stesso di BYOL/DINO)
        for (int i = 0; i < encoder.params().size(); i++)
            target_encoder.params()[i] = ema_decay * target_encoder.params()[i]
                                        + (1-ema_decay) * encoder.params()[i];
    }
};
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/world/world_model.h/.cpp` | Predictive World Model |
| `src/world/imagination.h/.cpp` | Simulazione multi-step nel latente |
| `src/world/jepa.h/.cpp` | JEPA: encoder + predictor + EMA target |
| `tests/test_world.cpp` | Predizione next-state, imagination |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_predict` | next_state ha shape corretto |
| `test_imagine` | Traiettoria di N step senza divergenza |
| `test_jepa_loss` | Loss decresce durante training su video sintetici |
| `test_jepa_ema` | Target encoder converge verso encoder online |
