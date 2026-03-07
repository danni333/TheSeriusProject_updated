# Fase 37 — Personalizzazione On-Device (Federated Learning)

> Il modello impara dai dati dell'utente senza violargli la privacy.
> Fine-tuning locale con adapter e aggregazione differenzialmente privata.

---

## 1. Local Differential Privacy

### Concetto

L'utente fa fine-tuning locale sul suo dispositivo. I delta dei pesi vengono **rumorizzati** prima di essere inviati al server per migliorare il modello globale.

```
Dispositivo utente:
  1. Fine-tune locale (pochi step LoRA)
  2. Calcola Δw = w_local - w_global
  3. Clip: ||Δw|| ≤ C
  4. Aggiungi rumore: Δw_noisy = Δw + N(0, σ²·C²·I)
  5. Invia Δw_noisy al server

Server:
  6. Aggrega: w_global += (1/N) · Σ Δw_noisy_i
```

```cpp
Tensor add_dp_noise(const Tensor& delta, float clip_norm, float sigma) {
    // 1. Clip la norma
    float norm = l2_norm(delta);
    Tensor clipped = (norm > clip_norm) ? delta * (clip_norm / norm) : delta;
    // 2. Aggiungi rumore gaussiano calibrato
    Tensor noise = Tensor::randn(delta.shape()) * (sigma * clip_norm);
    return clipped + noise;
}
```

### Privacy budget (ε, δ)

| σ (rumore) | ε (privacy) | Utilità |
|---|---|---|
| Alto (10.0) | 0.1 (molto privato) | Bassa |
| Medio (1.0) | 1.0 | Bilanciata |
| Basso (0.1) | 10.0 (poco privato) | Alta |

---

## 2. PEFT Adapter locale su Modelli Importati

Dal momento che Fase 15 carica comodamente un Large Language Model gigante pre-addestrato dalla rete locale (tramite GGUF mmap-ato *read-only*), il fine-tuning non necessita di memoria enorme: appiccicheremo dei sensori *LoRA* ai vari pesi (W_q, W_v) appoggiati in RAM/VRAM RW.

```cpp
class UserAdapter {
    string user_id;
    LoRAAdapter adapter;   // piccolo (~1-10MB)
    string local_path;     // checkpoint locale

    void personalize(Model& model, const vector<string>& user_data) {
        model.attach_adapter(adapter);
        // Fine-tune su dati locali dell'utente
        for (auto& text : user_data) {
            auto loss = model.forward_loss(tokenizer.encode(text));
            loss.backward();
            optimizer.step();
        }
        adapter.save(local_path);
    }

    void activate(Model& model) {
        adapter.load(local_path);
        model.attach_adapter(adapter);
    }
};
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/federated/dp_noise.h/.cpp` | Differential privacy: clip + noise |
| `src/federated/aggregator.h/.cpp` | Server-side aggregazione sicura |
| `src/federated/user_adapter.h/.cpp` | Adapter per-utente con LoRA |
| `tests/test_federated.cpp` | Privacy budget, aggregazione, adapter |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_dp_noise` | Norma clipped ≤ C, rumore aggiunto |
| `test_aggregation` | Media di N delta noisy ≈ delta reale |
| `test_adapter_personal` | Dopo fine-tune, modello migliora su dati utente |
| `test_privacy_budget` | ε calcolato correttamente per dato σ |
