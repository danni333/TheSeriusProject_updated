# Fase 44 — Quantum-Classical Hybrid Kernels

> Emulare circuiti quantistici in C++ per risolvere problemi
> di ottimizzazione dei pesi che le GPU classiche non possono gestire.

---

## 1. Emulatore di Circuiti Quantistici

### Rappresentazione dello stato

Un sistema di $n$ qubit ha uno stato $|\psi\rangle$ con $2^n$ ampiezze complesse.

```cpp
struct Qubit {
    complex<float> alpha;  // |0⟩ amplitude
    complex<float> beta;   // |1⟩ amplitude
};

class QuantumState {
    vector<complex<float>> amplitudes;  // 2^n valori
    int n_qubits;

    QuantumState(int n) : n_qubits(n), amplitudes(1 << n, {0,0}) {
        amplitudes[0] = {1, 0};  // |00...0⟩
    }

    // Applica gate a un qubit
    void apply_gate(const Tensor& gate, int qubit_idx);
    // Misura un qubit (collassa lo stato)
    int measure(int qubit_idx);
    // Probabilità di ogni stato
    vector<float> probabilities();
};
```

### Gate quantistici

```cpp
// Matrici 2x2 per gate standard
Tensor H = {{1/sqrt2, 1/sqrt2}, {1/sqrt2, -1/sqrt2}};   // Hadamard
Tensor X = {{0, 1}, {1, 0}};                              // NOT
Tensor RY(float theta) {                                   // Rotazione Y
    return {{cos(theta/2), -sin(theta/2)}, {sin(theta/2), cos(theta/2)}};
}
```

---

## 2. Variational Quantum Eigensolver (VQE) per Ottimizzazione

```cpp
class QuantumOptimizer {
    int n_qubits;
    int n_layers;

    // Circuito parametrizzato
    float evaluate(const vector<float>& params, const Tensor& cost_matrix) {
        QuantumState state(n_qubits);
        int p = 0;
        for (int l = 0; l < n_layers; l++) {
            for (int q = 0; q < n_qubits; q++)
                state.apply_gate(RY(params[p++]), q);
            for (int q = 0; q < n_qubits-1; q++)
                state.apply_cnot(q, q+1);
        }
        return expectation_value(state, cost_matrix);
    }

    // Ottimizza parametri classicamente
    vector<float> optimize(const Tensor& cost, int max_iter = 100) {
        vector<float> params = random_init(n_qubits * n_layers);
        for (int i = 0; i < max_iter; i++) {
            auto grads = parameter_shift_gradient(params, cost);
            for (int j = 0; j < params.size(); j++)
                params[j] -= lr * grads[j];
        }
        return params;
    }
};
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/quantum/state.h/.cpp` | Stato quantistico e gate |
| `src/quantum/circuit.h/.cpp` | Circuiti parametrizzati |
| `src/quantum/vqe.h/.cpp` | Variational optimizer |
| `tests/test_quantum.cpp` | Gate, misura, VQE convergenza |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_hadamard` | H|0⟩ = (|0⟩+|1⟩)/√2, probs = 50/50 |
| `test_entanglement` | CNOT crea stato entangled |
| `test_vqe` | Trova minimo di funzione costo semplice |
| `test_gradient` | Parameter shift gradient ≈ numerico |

