# Fase 19+ — Stato Implementazione e Cose Mancanti (LoRA)

Riferimento: `fase19_lora.md`

---

## 1. LoRA (Low-Rank Adaptation)

| Elemento | Stato | Nota |
|----------|--------|-----|
| Layer LoRA: $W' = W + \frac{\alpha}{r} (A \cdot B)$ | ✅ | Implementato in `src/lora/lora_linear.cpp/.h` con matrice $A$ (Kaiming) e $B$ (zero-init) |
| Solo A e B trainabili; W base congelato | ✅ | `LoRALinear::parameters()` restituisce solo i nodi autograd di $A$ e $B$ |
| Adapter multi-layer per transformer | ✅ | Implementato in `src/lora/lora_adapter.cpp/.h` con target configurabili |
| Salvataggio e caricamento file adapter binario | ✅ | Metodi `LoRAAdapter::save()` e `LoRAAdapter::load()` |
| Merge LoRA in W per export/inference pura | [/] | Forward combinato senza overhead; merge permanente per esportazione GGUF/Safetensors opzionale |

---

## 2. Training, Configurazione e Validazione

| Elemento | Stato | Nota |
|----------|--------|-----|
| Configurazione rank ($r$), alpha ($\alpha$) e dropout | ✅ | Supportata in `LoRAConfig` / costruttori di `LoRALinear` |
| Test B=0 garantisce output identico al base | ✅ | `LoRATest.InitBZeroMeansNoChange` in `tests/test_lora.cpp` |
| Test autograd e backward su A e B | ✅ | `LoRATest.BackwardGradientsFlowToAB` |
| Test serializzazione round-trip adapter | ✅ | `LoRATest.SaveAndLoadAdapter` |
| Test scaling $\alpha / r$ | ✅ | `LoRATest.AlphaScaling` |

---

## 3. File sorgente presenti (fase 19)

| File | Stato | Descrizione |
|------|--------|-------------|
| `src/lora/lora_linear.h/.cpp` | ✅ | Layer lineare parametrizzato a basso rango |
| `src/lora/lora_adapter.h/.cpp` | ✅ | Gestore di adapter per l'intero modello e persistenza |
| `tests/test_lora.cpp` | ✅ | Suite di test esaustiva (shape, autograd, forward, serializzazione) |

---

**Priorità rimanenti:** utility di esportazione di pesi fusi (`merge_and_unload`) verso formati Safetensors/GGUF per deployment ad alte prestazioni.
