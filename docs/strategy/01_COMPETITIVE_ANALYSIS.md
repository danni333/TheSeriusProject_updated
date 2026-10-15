# 📊 Analisi Competitiva — Stato dell'Arte AI (Marzo 2026)

> Questo documento analizza in dettaglio ogni famiglia di modelli AI di frontiera,
> identificando le innovazioni chiave che Serius deve incorporare e superare.

---

## 1. DeepSeek (Cina — DeepSeek AI)

### Modelli Principali
| Modello | Release | Params | Attivi/Token | Focus |
|---|---|---|---|---|
| DeepSeek V3 | Dic 2024 | 671B | 37B | General-purpose |
| DeepSeek R1 | Gen 2025 | 671B | 37B | Reasoning (math, code, logic) |
| DeepSeek V3-0324 | Mar 2025 | 671B | 37B | Reasoning + tool-use migliorati |
| DeepSeek R1-0528 | Mag 2025 | 671B | 37B | Reasoning estremo |
| DeepSeek V3.1 | Ago 2025 | 671B | 37B | Hybrid thinking/non-thinking |
| DeepSeek V3.2 | Dic 2025 | 671B | 37B | DeepSeek Sparse Attention (DSA) |

### Innovazioni Architetturali

#### Multi-head Latent Attention (MLA)
La MLA è l'innovazione più impattante di DeepSeek. Invece di memorizzare separatamente K e V per ogni head nel KV-cache, MLA comprime K e V in uno **spazio latente a basso rango**:

```
Standard GQA:  KV-cache = n_kv_heads × seq_len × head_dim × 2
MLA:           KV-cache = seq_len × latent_dim  (dove latent_dim << n_heads × head_dim)
```

**Riduzione KV-cache: 4-8x** mantenendo la stessa qualità.

**Come funziona:**
1. Input → Proiezione down in spazio latente (d_model → d_latent)
2. Lo spazio latente viene cached (tiny!)
3. Al momento dell'attenzione, ri-proietta up: d_latent → K, V separati per ogni head
4. Il costo computazionale dell'up-projection è trascurabile confrontato col risparmio di memoria

#### Mixture-of-Experts (MoE) — 256 Esperti
- **256 routed experts** + 1 shared expert always-on
- Solo **8 esperti attivati per token** (3.1% di attivazione)
- Load balancing tramite auxiliary loss per evitare che pochi esperti dominano
- Risultato: 671B params totali ma costo di inference come un modello ~40B denso

#### Multi-Token Prediction (MTP)
- Il modello predice **2 token nel futuro** in un singolo forward pass
- Usato sia in training (objective aggiuntivo) che in inference (1.8x speedup)
- In inference: verifica il secondo token in parallelo, se corretto lo accetta

#### FP8 Pre-training
- Primo grande modello open-source trainato in **FP8 precision**
- Riduzione costo training ~2x senza degradazione di qualità
- Richiede careful loss scaling e gradient handling

#### GRPO (Group Relative Policy Optimization)
- RL senza reward model separato (costa troppo trainarne uno)
- Genera N risposte, confronta tra loro, usa la migliore come reward
- Rule-based verification per math e coding (esegue il codice!)

### Lezioni per Serius
> - **MLA è must-have** — il KV-cache è il bottleneck #1 per long-context
> - **MoE a 256 esperti è aggressivo** — loro ci sono arrivati gradualmente (V2: 160, V3: 256)
> - **FP8 training è il futuro** — noi abbiamo già FP8 per inference, serve per training
> - **GRPO è elegante** — nessun reward model = molto più facile da implementare

---

## 2. Minimax (Cina — MiniMax)

### Modelli Principali
| Modello | Release | Params | Attivi/Token | Focus |
|---|---|---|---|---|
| Minimax Text-01 | Inizio 2025 | 456B | ~46B | Hybrid linear attention |
| Minimax-M1 | 2025 | 456B | 45.9B | Open-weight reasoning |
| Minimax-M2 | 2025 | 230B | 10B | Cost-effective MoE |
| Minimax-M2.1 | 2026 | ~230B | 10B | Coding + agentic workflows |
| Minimax-M2.5 | 2025 | 80B | 20B | Multilingual, 200K context |
| Minimax-M2.7 | Mar 2026 | N/D | N/D | Self-evolving reasoning |

### Innovazioni Chiave

#### Lightning Attention
- **75% riduzione FLOPs** rispetto a DeepSeek R1 su sequenze di 100K tokens
- Architettura 1:7 di attenzione ibrida: 1 layer di full attention ogni 7 di linear attention
- La linear attention usa approssimazione kernel (Random Feature Maps)
- Il risultato: context window di **1M tokens** con costo quasi-lineare

#### CISPO (Custom RL Pipeline)
- Variante custom del RL training pipeline
- Dettagli non completamente pubblici, ma risultati superiori a GRPO su alcuni benchmark
- Ottimizzato per reasoning multi-step

#### Self-Evolving (M2.7)
- Il modello **esegue 30-50% del workflow di RL research** autonomamente
- Genera i propri dati di training, li valuta, si re-trena
- Loop di auto-miglioramento senza intervento umano

#### Plan-Act-Verify (M2)
- Pattern agentico ottimizzato per M2 con soli 10B attivi
- Il modello: (1) Pianifica cosa fare, (2) Esegue l'azione, (3) Verifica il risultato
- Loop iterativo fino a completamento del task

### Lezioni per Serius
> - **Lightning Attention è rivoluzionaria** — sub-quadratic attention che funziona davvero
> - **Self-evolving è il futuro** — il modello che si migliora da solo è il goal ultimo
> - **10B attivi è sufficiente** — M2 compete con modelli 10x più grandi

---

## 3. NVIDIA Nemotron 3 (USA — NVIDIA)

### Famiglia Nemotron 3
| Variante | Params | Attivi | Target |
|---|---|---|---|
| Nemotron 3 Nano | ~8B | ~2B | Edge, on-device |
| Nemotron 3 Super | 120B | ~12B | Production agentic AI |
| Nemotron 3 Ultra | 300B+ | ~30B | Frontier research |

### Innovazioni Chiave

#### Hybrid Mamba-Transformer
Questa è l'architettura più innovativa del 2025-2026:

```
Layer composition (Super):
  Mamba-2 layers: ~75%  (sequence processing, lineare O(n))
  Attention layers: ~25% (retrieval, quadratica O(n²) ma solo ogni 4-8 layers)
```

**Perché funziona:**
- **Mamba-2 (SSM)**: processa sequenze in tempo lineare, perfetto per il 90% dei token
- **Attention**: necessaria per retrieval accurato da contesti lontani (finding needles in haystacks)
- Il mix 75/25 cattura il meglio di entrambi i mondi

#### LatentMoE
Evoluzione del MoE standard:
1. Token → proiezione in **spazio latente compresso** (es. d_model → d_latent, 4x riduzione)
2. Il router opera sullo spazio latente (più veloce, meno parametri)
3. Routing a **4x più esperti** per lo stesso costo computazionale
4. Risultato: specializzazione più granulare degli esperti

```
MoE standard:  Token(d=4096) → Router → Expert(d=4096)
LatentMoE:     Token(d=4096) → Compress(d=1024) → Router → Expert(d=1024) → Expand(d=4096)
```

#### NVFP4 Pre-training
- Training nativo in **4-bit floating point** (non post-training quantization!)
- Formato custom NVIDIA ottimizzato per Blackwell GPUs
- **4x faster inference** su Blackwell vs FP8 su Hopper
- Zero loss di accuracy rispetto a FP16

#### Multi-Token Prediction
- Come DeepSeek, predice 2+ token per forward pass
- Integrato con speculative decoding per massima velocità

### Lezioni per Serius
> - **Hybrid Mamba-Transformer è la direzione** — NVIDIA ci scommette con i suoi chip
> - **LatentMoE è superiore al MoE standard** — compressione prima del routing = win
> - **NVFP4 richiede hardware specializzato** — noi possiamo emularlo via software
> - **1M token context** con costo sub-quadratico = combinazione Mamba + sparse attention

---

## 4. Anthropic Claude (USA — Anthropic)

### Modelli (Gen 4+)
| Modello | Release | Context | Focus |
|---|---|---|---|
| Claude Opus 4 | Mag 2025 | 200K | Top-tier intelligence |
| Claude Sonnet 4 | Mag 2025 | 200K | Balance perf/cost |
| Claude Haiku 4.5 | Ott 2025 | 200K | Speed + cost efficiency |
| Claude Opus 4.6 | Feb 2026 | **1M beta** | SWE + agentic best-in-class |
| Claude Sonnet 4.6 | Feb 2026 | **1M beta** | Opus-level at Sonnet price |

### Innovazioni Chiave

#### Hybrid Reasoning Core
- Switch dinamico tra "quick answer" e "deep thinking"
- Extended thinking: il modello usa token interni per ragionamento step-by-step
- Budget computazionale variabile: problemi facili → risposta veloce, problemi hard → più compute

#### Constitutional AI
- Allineamento tramite principi scritti (la "costituzione")
- AI che si auto-corregge basandosi su principi etici
- Riduce la necessità di human feedback costoso (RLHF)

#### Agentic Capabilities (Opus 4.6)
- **Terminal-Bench 2.0: 65.4%** (best-in-class per SWE tasks)
- **OSWorld: 72.7%** (best per computer use)
- Può operare autonomamente per ore su codebase grandi
- Computer use: controlla desktop, browser, terminale

### Lezioni per Serius
> - **Il thinking mode è essenziale** — Claude dimostra che switch dinamico tra fast/slow è vincente
> - **1M context** è ora lo standard minimo per modelli frontier
> - **Agentic = computer use** — il modello non solo genera testo, opera nel mondo reale
> - **Constitutional AI** è un approccio elegante per safety senza RLHF costoso

---

## 5. Alibaba Qwen (Cina — Alibaba Cloud)

### Famiglia Qwen
| Modello | Release | Params | Attivi | Focus |
|---|---|---|---|---|
| Qwen 2.5 (dense) | Gen 2025 | 0.5-72B | 100% | General-purpose |
| Qwen2.5-Max (MoE) | Gen 2025 | N/D | N/D | Large-scale reasoning |
| Qwen 3 | Apr 2025 | 0.6-235B | 22B (MoE) | Thinking/non-thinking |
| Qwen3-Next | Set 2025 | 80B | 3B | Long-context champion |
| Qwen 3.5 | Feb 2026 | 397B | 17B | Native multimodal agent |

### Innovazioni Chiave

#### Gated DeltaNet
Alternativa ai SSM (Mamba) per il layer di sequenza:
- Usa **gating mechanism** per controllare il flusso di informazione
- Combinato con sparse MoE e attention tradizionale
- Efficienza simile a Mamba ma con formazione più stabile

#### Thinking / Non-Thinking Mode (Qwen 3+)
- **Thinking mode**: chain-of-thought interno, reasoning multi-step
- **Non-thinking mode**: risposta diretta, bassa latenza
- Il modello decide autonomamente quale mode usare basandosi sulla query

#### 128 Esperti, 8 Attivi (Qwen 3 MoE)
- Nessun "shared expert" (a differenza di DeepSeek)
- Router top-k con k=8
- Risultato: forte specializzazione per domain

#### Native Multimodal Agent (Qwen 3.5)
- Vision + text + audio in un unico backbone unificato
- Non è un vision encoder aggiunto dopo: la multimodalità è nativa dall'inizio
- **201 lingue e dialetti** supportati
- **262K context nativo** (1M in versione hosted)

### Lezioni per Serius
> - **DeltaNet è un'alternativa seria a Mamba** — diversificare le scelte architetturali
> - **Thinking mode è universale** — tutti i top models lo hanno adottato
> - **Multimodalità nativa > bolt-on** — pianificare embedding unificato dall'inizio
> - **L'apertura paga** — Qwen è open-weight e questo accelera l'adozione

---

## 6. Meta Llama 4 (USA — Meta)

### Modelli Llama 4
| Modello | Release | Params | Attivi | Context | Esperti |
|---|---|---|---|---|---|
| Llama 4 Scout | Apr 2025 | 109B | 17B | **10M tokens** | 16 |
| Llama 4 Maverick | Apr 2025 | 400B | 17B | 1M | 128 |
| Llama 4 Behemoth | In training | ~2T | ~200B | N/D | N/D |

### Innovazioni Chiave

#### 10 Million Token Context (Scout)
- Record assoluto per context window
- Usa combinazione di RoPE scaling + sparse attention + memory-efficient encoding
- Funziona su **un singolo H100 con INT4 quantization**

#### Early Fusion Multimodal
- Testo e immagini trattati in modo unificato dall'inizio del training
- Non è un vision encoder bolted-on: i token di immagine sono first-class citizens
- Risultato: comprensione cross-modal più profonda

#### Knowledge Distillation da Behemoth
- Behemoth (2T params) usato come teacher
- Scout (109B) e Maverick (400B) distillati da Behemoth
- Trasferimento di "intelligenza" dal modello impossibile da deployare (2T) a modelli pratici

### Lezioni per Serius
> - **10M context è il nuevo benchmark** — la gara sui context window non si ferma
> - **Knowledge distillation è fondamentale** — trena un mega-modello, distilla in modelli pratici
> - **Early fusion multimodal** è superiore al late-fusion approach
> - **Un singolo H100** dovrebbe bastare per inference di modelli fino a 100B con quant

---

## 7. Mistral AI (Francia)

### Modelli
| Modello | Release | Params | Attivi | Focus |
|---|---|---|---|---|
| Mistral Large 3 | Dic 2025 | 675B | 41B | Frontier MoE |
| Mistral Small 4 | **Mar 2026** | 119B | 6B | Unified model |
| Devstral Small 2 | Dic 2025 | 24B | 24B | Coding |
| Magistral Medium | Giu 2025 | N/D | N/D | Chain-of-thought |

### Innovazione Chiave: Unified Model (Small 4)
- **Un solo modello per tutto**: reasoning (Magistral) + vision (Pixtral) + coding (Devstral)
- 119B total, solo **6B attivi** → estremamente efficiente
- **256K context window**
- Apache 2.0 license → completamente utilizzabile

### Lezioni per Serius
> - **Unificazione è la tendenza** — basta con modelli separati per ogni task
> - **6B attivi competono con 30B+ densi** — l'efficienza MoE è reale
> - **Apache 2.0** dimostra che open-source funziona come business model

---

## 8. Microsoft Phi-4 (USA)

### Modelli
| Modello | Params | Focus |
|---|---|---|
| Phi-4 | 14B (dense) | Reasoning, math |
| Phi-4-mini | 3.8B | Mobile, edge |
| Phi-4-multimodal | 5.6B | Text + image + audio + video |
| Phi-4-reasoning | 14B | RL-enhanced chain-of-thought |

### Innovazione Chiave: Quality Over Quantity
- **Synthetic data >> raw internet data** per training di modelli piccoli
- Phi-4-mini (3.8B) batte modelli 7-9B su ragionamento
- Training su dati generati da modelli più grandi, curati e filtrati
- Risultato: performance/param ratio eccezionale

### Lezioni per Serius
> - **Dati sintetici di qualità** possono compensare parametri mancanti
> - **3.8B è sufficiente** per task di ragionamento su edge
> - **Phi prova che C++ nativo + modelli piccoli = deployment ovunque**

---

## 9. Google Gemma 3 (USA)

### Modelli
| Modello | Params | Focus | Target |
|---|---|---|---|
| Gemma 3 27B | 27B (dense) | General-purpose | Single GPU |
| Gemma 3 4B | 4B | Lightweight | Laptop/phone |
| Gemma 3 1B | 1B | Ultra-light | IoT, embedded |
| Gemma 3 270M | 270M | Hyper-efficient | Smartwatch, MCU |
| Gemma 3n | Variable | On-device multimodal | Phones, tablets |

### Innovazione Chiave: Interleaved Local/Global Attention
```
Layer 1: Local Attention  (sliding window, low KV-cache)
Layer 2: Local Attention
Layer 3: Local Attention
Layer 4: Global Attention (full context, high KV-cache)
Layer 5: Local Attention
...repeat
```

- 3/4 dei layers usano **local attention** → KV-cache ridotto
- 1/4 usa **global attention** → recall da contesti lontani preservato
- Risultato: 128K context window con memoria gestibile

#### Quantization-Aware Training (QAT)
- Il modello è trainato **sapendo che verrà quantizzato**
- Risultato: INT4 con quasi zero loss di qualità (vs post-training quant che perde)
- Gemma 3 27B gira su una **consumer GPU** con QAT + INT4

### Lezioni per Serius
> - **Interleaved attention** è una soluzione elegante per il memory problem
> - **QAT è superiore** al post-training quantization — integrarlo nel training loop
> - **270M params funzionanti** = c'è mercato per modelli ultra-piccoli
> - **Edge-first design** è il differenziatore più forte per un framework C++

---

## Sintesi: Mappa delle Innovazioni

```
                    MoE    MLA    Mamba   MTP   RL/GRPO   Multimodal   Long-Context
DeepSeek V3.2       ████   ████   ░░░░   ████   ████      ░░░░         ████ (128K)
Minimax M2.7        ████   ░░░░   ░░░░   ░░░░   ████      ░░░░         ████ (1M)  
Nemotron 3          ████   ░░░░   ████   ████   ░░░░      ░░░░         ████ (1M)
Claude 4.6          ░░░░   ░░░░   ░░░░   ░░░░   ████      ████         ████ (1M)
Qwen 3.5            ████   ░░░░   ████*  ░░░░   ░░░░      ████         ████ (262K)
Llama 4             ████   ░░░░   ░░░░   ░░░░   ░░░░      ████         ████ (10M!)
Mistral Small 4     ████   ░░░░   ░░░░   ░░░░   ████      ████         ████ (256K)
Phi-4               ░░░░   ░░░░   ░░░░   ░░░░   ████      ████         ░░░░
Gemma 3             ░░░░   ░░░░   ░░░░   ░░░░   ░░░░      ████         ████ (128K)

████ = implementato    ░░░░ = non implementato    * = Gated DeltaNet (alternativa SSM)

SERIUS TARGET:      ████   ████   ████   ████   ████      ████         ████ (1M+)
                    TUTTI. Nessun competitor li ha tutti.
```

---

> **Conclusione**: Nessun singolo competitor implementa TUTTE le innovazioni contemporaneamente.
> DeepSeek ha MLA ma non Mamba. Nemotron ha Mamba ma non MLA. Nessuno ha MLA + Mamba + MoE + MTP + RL.
> **Questa è la finestra di opportunità di Serius: essere il primo a combinarle tutte.**
