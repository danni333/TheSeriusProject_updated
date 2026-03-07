# Fase 2 — Autograd (Differenziazione Automatica)

> Per addestrare un modello servono i **gradienti**.
> Autograd registra ogni operazione in un **grafo computazionale**
> e può percorrerlo al contrario per calcolare automaticamente le derivate.

---

## 1. Computational Graph

### Obiettivo

Ogni operazione matematica (somma, matmul, softmax…) crea un **nodo** in un grafo diretto aciclico (DAG). Questo grafo viene poi percorso al contrario per calcolare i gradienti.

### Struttura del nodo

| Campo | Tipo | Descrizione |
|---|---|---|
| `data` | `Tensor` | Valore del tensore (risultato forward) |
| `grad` | `Tensor` | Gradiente accumulato (popolato in backward) |
| `children` | `vector<Node*>` | Nodi di input che hanno prodotto questo nodo |
| `backward_fn` | `function<void()>` | Funzione che calcola `∂L/∂input` per ogni child |
| `requires_grad` | `bool` | Se `true`, partecipa alla backpropagation |

### Flusso

```
┌─────────┐     ┌─────────┐
│  A (leaf)│────▶│         │
└─────────┘     │  C=A*B  │────▶ ... ────▶ Loss (scalare)
┌─────────┐     │         │
│  B (leaf)│────▶│         │
└─────────┘     └─────────┘

Forward:  A, B → C → ... → Loss
Backward: Loss → ... → C → A.grad, B.grad
```

### Operazioni che creano nodi

Ogni operazione wrappa il calcolo forward e registra la `backward_fn`:

```cpp
Node* matmul(Node* A, Node* B) {
    Node* C = new Node(blas::matmul(A->data, B->data));
    C->children = {A, B};
    C->backward_fn = [=]() {
        // ∂L/∂A = ∂L/∂C * Bᵀ
        A->grad += blas::matmul(C->grad, transpose(B->data));
        // ∂L/∂B = Aᵀ * ∂L/∂C
        B->grad += blas::matmul(transpose(A->data), C->grad);
    };
    return C;
}
```

---

## 2. Backpropagation

### Algoritmo

1. **Forward pass**: calcola il valore di ogni nodo dalla foglia alla loss.
2. **Topological sort**: ordina i nodi così che ogni nodo viene dopo i suoi figli.
3. **Backward pass**: percorri l'ordine inverso, chiama `backward_fn()` su ogni nodo.

```
backward(loss_node):
    loss_node.grad = Tensor::ones(loss_node.shape)   // ∂L/∂L = 1
    topo_order = topological_sort(loss_node)
    for node in reversed(topo_order):
        if node.backward_fn:
            node.backward_fn()
```

### Accumulo gradienti

I gradienti si **accumulano** (`+=`), non si sovrascrivono. Questo è necessario quando un tensore è usato in più operazioni. Prima di ogni nuovo forward pass, si chiama `zero_grad()` su tutti i parametri.

### Topological Sort

```cpp
void topo_sort(Node* node, set<Node*>& visited, vector<Node*>& order) {
    if (visited.count(node)) return;
    visited.insert(node);
    for (auto* child : node->children)
        topo_sort(child, visited, order);
    order.push_back(node);
}
```

---

## 3. Funzioni di Attivazione (Forward + Backward)

### ReLU

| | Formula |
|---|---|
| **Forward** | $\text{ReLU}(x) = \max(0, x)$ |
| **Backward** | $\frac{\partial}{\partial x} = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ |

### GELU (Gaussian Error Linear Unit)

Usata nei Transformer classici (GPT-2, BERT). I modelli di frontiera più recenti (LLaMA, Mistral, DeepSeek) preferiscono **SwiGLU** (`SiLU(x) * Linear(x)`) che offre migliore espressività.

| | Formula |
|---|---|
| **Forward** | $\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)$ |
| **Backward** | Derivata dell'approssimazione tanh (vedi implementazione) |

### Softmax

| | Formula |
|---|---|
| **Forward** | $\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$ |
| **Backward** | $\frac{\partial L}{\partial x_i} = s_i \left(\frac{\partial L}{\partial s_i} - \sum_j s_j \frac{\partial L}{\partial s_j}\right)$ |

> [!IMPORTANT]
> Sottrai `max(x)` prima dell'esponenziale per **stabilità numerica**.
> Senza questo trick, `exp(x)` va in overflow per valori grandi.

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `src/autograd/node.h` | Classe `Node` (data, grad, backward_fn, children) |
| `src/autograd/node.cpp` | Implementazione |
| `src/autograd/engine.h` | `backward()`, topological sort |
| `src/autograd/engine.cpp` | Implementazione |
| `src/autograd/ops.h` | Operazioni differenziabili (matmul, add, relu, gelu, softmax) |
| `src/autograd/ops.cpp` | Implementazione forward + backward di ogni op |
| `tests/test_autograd.cpp` | Unit test con gradient checking |

---

## 5. Piano di Verifica

### Gradient Checking Numerico

Per verificare che i gradienti analitici siano corretti, confrontarli con la derivata numerica:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

con $\epsilon = 10^{-5}$. Se la differenza relativa è $< 10^{-5}$, il gradiente è corretto.

### Test da eseguire

| Test | Verifica |
|---|---|
| `test_relu_grad` | Gradiente ReLU: 1 per x>0, 0 per x≤0 |
| `test_gelu_grad` | Gradiente GELU vs. numerico |
| `test_softmax_grad` | Gradiente Softmax vs. numerico |
| `test_matmul_grad` | Gradienti ∂L/∂A e ∂L/∂B nella matmul |
| `test_chain` | Catena di operazioni: verifica backprop end-to-end |
| `test_shared_input` | Un tensore usato due volte: gradienti si accumulano |
