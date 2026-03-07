# Fase 42 — Liquid Neural Networks & CfC (Closed-form Continuous Time)

> Il modello non vede il tempo come sequenza discreta (1, 2, 3)
> ma come un flusso continuo. I parametri cambiano durante l'elaborazione.

---

## 1. ODE-based Layers

### Concetto

Invece di layer fissi $h_{t+1} = f(h_t)$, lo stato evolve secondo un'equazione differenziale:

$$\frac{dh}{dt} = f_\theta(h(t), x(t), t)$$

### Neural ODE

```cpp
class NeuralODE {
    NeuralNet f;  // rete che definisce la derivata

    Tensor forward(const Tensor& h0, float t0, float t1, int steps) {
        Tensor h = h0;
        float dt = (t1 - t0) / steps;
        // Runge-Kutta 4° ordine
        for (int i = 0; i < steps; i++) {
            float t = t0 + i * dt;
            Tensor k1 = f(h, t);
            Tensor k2 = f(h + 0.5f*dt*k1, t + 0.5f*dt);
            Tensor k3 = f(h + 0.5f*dt*k2, t + 0.5f*dt);
            Tensor k4 = f(h + dt*k3, t + dt);
            h = h + (dt/6.0f) * (k1 + 2*k2 + 2*k3 + k4);
        }
        return h;
    }
};
```

---

## 2. Liquid Time-Constant (LTC) Networks

### Neuroni con dinamica temporale variabile

```cpp
struct LTCNeuron {
    // τ(x) varia con l'input → adattamento alla complessità
    Tensor forward(const Tensor& x, const Tensor& h, float dt) {
        Tensor tau = sigmoid(W_tau * x + b_tau);  // time constant adattivo
        Tensor f_val = tanh(W_h * h + W_x * x + b);
        // Update continuo
        return h + dt * (1.0f / tau) * (f_val - h);
    }
};
```

### Adattamento alla complessità

```
Input semplice "Ciao" → τ grande → pochi step → risposta rapida
Input complesso "Dimostra il Teorema di Fermat" → τ piccolo → molti step → elaborazione profonda
```

```cpp
int adaptive_steps(const Tensor& x, float tolerance) {
    // Più è complesso l'input, più step servono
    Tensor h = initial_state;
    for (int steps = 1; steps <= max_steps; steps++) {
        Tensor h_new = ltc.forward(x, h, 1.0f / steps);
        if (l2_norm(h_new - h) < tolerance)
            return steps;  // convergenza raggiunta
        h = h_new;
    }
    return max_steps;
}
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/liquid/neural_ode.h/.cpp` | Neural ODE con RK4 |
| `src/liquid/ltc.h/.cpp` | Liquid Time-Constant neurons |
| `src/liquid/adaptive.h/.cpp` | Step adattivi basati su complessità |
| `tests/test_liquid.cpp` | ODE converge, LTC si adatta |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_ode_convergence` | Più step RK4 → soluzione più precisa |
| `test_ltc_adaptive` | Input semplice → meno step; complesso → più step |
| `test_ltc_vs_lstm` | Qualità comparabile su sequence modeling |
| `test_continuous_time` | Funziona con intervalli temporali irregolari |
