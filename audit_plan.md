# Audit Completo — Piano di Lavoro

**Data:** 2026-11-23
**Obiettivo:** trovare bug reali (non stile) in tutto il motore, partendo dalle aree ad alto blast radius.
**Metodo:** una fase alla volta, lettura mirata file per file, findings con file:linea e severità. Alla fine: report consolidato + build/ctest di verifica.

## Fasi (ordine = rischio decrescente)

| # | Area | File chiave | Cosa cercare |
|---|------|-------------|--------------|
| 0 | Piano | questo file | — |
| 1 | **Tensor + Memory lifetime** | `src/tensor/tensor.{h,cpp}`, `src/memory/*` | use-after-free, double-free, view/owner confusione, sync host/device, stride/reshape edge case |
| 2 | **Autograd** | `src/autograd/*` | leak dei Node, gradient errati, backward incompleti, topological order |
| 3 | **BLAS + dispatch + quant** | `src/blas/*`, `src/quant/*` | race su buffer, bound check, numerica (dequant, int4/fp8), dispatch sbagliato |
| 4 | **Transformer + inference** | `src/transformer/*`, `src/infer/llama_inference*`, `src/infer/sampler.cpp` | KV cache, RoPE, shape errate, sampling edge case |
| 5 | **Server + CLI + concorrenza** | `src/server/*`, `src/cli/*`, `src/stress/*` | data race, JSON parsing, HTTP edge case, prompt injection nel log |
| 6 | **Tokenizer + IO + GGUF** | `src/tokenizer/*`, `src/io/*` | off-by-one BPE, merge errati, bounds del loader, endian |
| 7 | **Resto** | `src/hub`, `distributed`, `safety`, `agent`, `rag`, `moe`, `lora`, `mcp`, ecc. | stub che mentono, catch silenziosi, TODO critici |
| 8 | **Verifica** | build + `ctest --output-on-failure` | regression, test che falliscono |

## Output

- Findings man mano in `audit_findings.md` (creato alla fine della fase 1)
- Report finale consolidato in `audit_report.md`

## Regole

- Un finding = file:linea, descrizione, severità (CRITICO/ALTO/MEDIO/BASSO), perché è un bug
- Non fixare niente durante l'audit — prima mappiamo tutto
- Se una fase esplode, la suddivido ulteriormente

## Stato avanzamento

- [x] Fase 1 — Tensor + Memory (17 finding)
- [x] Fase 2 — Autograd (15 finding)
- [x] Fase 3 — BLAS + quant (13 finding)
- [x] Fase 4 — Transformer + inference (32 finding; CUDA+CPU+offload+speculative coperti)
- [x] Fase 5 — Server + CLI (21 finding)
- [x] Fase 6 — Tokenizer + IO
- [ ] Fase 7 — Resto
- [ ] Fase 8 — Build + test
