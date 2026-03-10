# Fase 2+ — Cose mancanti (Autograd)

Riferimento: `fase2_autograd.md`

---

## 1. Computational graph e Node

| Elemento | Stato | Nota |
|----------|--------|-----|
| Nodo: data, grad, children, backward_fn, requires_grad | Da verificare | node.h / node.cpp |
| Operazioni che creano nodi (matmul, add, relu, gelu, softmax) | ✅ | ops.h / ops.cpp |
| Topological sort e backward pass | ✅ | engine.h / engine.cpp |
| Accumulo gradienti (+=), zero_grad | Da verificare | |
| Gradient checking numerico (ε=1e-5) | Da verificare in test | |

---

## 2. Funzioni di attivazione (forward + backward)

| Op | Stato |
|----|--------|
| ReLU (forward + backward) | Da verificare in ops |
| GELU (forward + backward, approssimazione tanh) | Da verificare |
| Softmax (con max subtraction per stabilità) | Da verificare |
| SwiGLU (citato in fase3 come preferito) | Opzionale per Fase2 |

---

## 3. Struttura file

| File | Stato |
|------|--------|
| src/autograd/node.h, .cpp | ✅ |
| src/autograd/engine.h, .cpp | ✅ |
| src/autograd/ops.h, .cpp | ✅ |
| tests/test_autograd.cpp | ✅ |

---

## 4. Test da avere (fase2)

| Test | Stato |
|------|--------|
| test_relu_grad | Da verificare |
| test_gelu_grad (vs numerico) | Da verificare |
| test_softmax_grad (vs numerico) | Da verificare |
| test_matmul_grad (∂L/∂A, ∂L/∂B) | Da verificare |
| test_chain (backprop end-to-end) | Da verificare |
| test_shared_input (accumulo gradienti) | Da verificare |

---

**Priorità:** assicurare che tutti i test di gradient checking siano presenti e passino; documentare tolleranze (FP32 1e-5).
