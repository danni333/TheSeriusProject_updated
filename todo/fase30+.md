# Fase 30+ — Cose mancanti (Kernel fusion)

Riferimento: `fase30_kernel_fusion.md`

---

## 1. Kernel fusion

| Elemento | Stato | Nota |
|----------|--------|-----|
| Fused kernel CUDA (es. Linear+ReLU, LayerNorm+Residual) | Parziale | fused_kernels.cu presente |
| Riduzione lanci kernel e accessi memoria | Da verificare | |
| Pattern: matmul+add, layernorm+residual, attention+dropout | Da verificare | |
| test_fused_cuda.cpp | ✅ | |
| Documentazione pattern in AUDIT | ✅ | fase30 e AUDIT_IMPLEMENTATION_RECOMMENDATIONS |

---

## 2. CPU

| Elemento | Stato |
|----------|--------|
| Fusion elementwise in un loop (es. add+relu) | Da verificare in blas/elementwise |
| Macro-op in autograd (fuse backward) | ❌ |

---

**Priorità:** misurare speedup fused vs non-fused; aggiungere pattern mancanti.
