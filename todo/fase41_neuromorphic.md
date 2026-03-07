# Fase 41 — Neuromorphic Computing & Asynchronous Processing

> Il cervello non è sincrono. I neuroni sparano "spike" solo quando serve.
> SNN e Event-Driven Execution per risparmio energetico del 90%.

---

## 1. Spiking Neural Networks (SNN)

### Differenza fondamentale

| | ANN (Transformer) | SNN (Neuromorphic) |
|---|---|---|
| Segnale | Float continui | Spike discreti (0/1) nel tempo |
| Calcolo | Ogni layer, ogni step | Solo quando un neurone "spara" |
| Energia | Costante | Proporzionale all'attività |

### Modello Leaky Integrate-and-Fire (LIF)

```cpp
struct LIFNeuron {
    float membrane_potential = 0;
    float threshold = 1.0f;
    float decay = 0.9f;       // leak factor
    float reset_value = 0;

    bool step(float input_current) {
        // Integra
        membrane_potential = decay * membrane_potential + input_current;
        // Spara?
        if (membrane_potential >= threshold) {
            membrane_potential = reset_value;
            return true;  // SPIKE!
        }
        return false;  // silenzio
    }
};
```

### Layer SNN

```cpp
class SpikingLinear {
    Tensor weights;  // (out, in)
    vector<LIFNeuron> neurons;

    vector<bool> forward(const vector<bool>& input_spikes) {
        vector<bool> output(neurons.size());
        for (int i = 0; i < neurons.size(); i++) {
            float current = 0;
            for (int j = 0; j < input_spikes.size(); j++)
                if (input_spikes[j])  // solo spike attivi
                    current += weights(i, j);
            output[i] = neurons[i].step(current);
        }
        return output;
    }
};
```

---

## 2. Event-Driven Execution

### Sparsità naturale degli spike

```
Timestamp  Neurone attivi (su 1000):
t=0:       [3, 15, 42, 178]              → 4/1000 = 0.4% attivi
t=1:       [7, 42, 301, 555, 890]        → 5/1000 = 0.5% attivi
t=2:       [15, 42]                       → 2/1000 = 0.2% attivi
```

→ Calcoli solo sui neuroni attivi: **99.5% del lavoro risparmiato**.

### Kernel event-driven

```cpp
void event_driven_forward(const vector<int>& active_neurons,
                           const Tensor& weights, vector<LIFNeuron>& neurons,
                           vector<int>& next_active) {
    // Calcola solo le connessioni dei neuroni attivi
    for (int src : active_neurons) {
        for (int dst = 0; dst < neurons.size(); dst++) {
            float w = weights(dst, src);
            if (fabsf(w) < 1e-6f) continue;  // skip connessioni deboli
            neurons[dst].membrane_potential += w;
        }
    }
    // Controlla quali neuroni sparano
    for (int i = 0; i < neurons.size(); i++)
        if (neurons[i].check_and_fire())
            next_active.push_back(i);
}
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/neuro/lif_neuron.h/.cpp` | Modello LIF (Leaky Integrate-and-Fire) |
| `src/neuro/spiking_layer.h/.cpp` | Layer SNN con connessioni pesate |
| `src/neuro/event_engine.h/.cpp` | Esecuzione event-driven |
| `src/neuro/encoder.h/.cpp` | Rate/temporal encoding (float → spike) |
| `tests/test_snn.cpp` | LIF dynamics, spike propagation |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_lif_threshold` | Neurone spara solo sopra soglia |
| `test_lif_decay` | Potenziale decade senza input |
| `test_event_sparse` | Attività < 5% dei neuroni per step |
| `test_snn_xor` | SNN apprende XOR con STDP |

