# 🎯 FASE 4 — Reasoning & Reinforcement Learning
**Periodo**: Q1 2027 (Gennaio - Marzo) | **Tasks**: 20 | **Completate**: 0

---

## 4.1 GRPO Pipeline

### T4.01 — Batch Response Generation
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/rl/batch_generator.h` (NUOVO)
- **Criteria**: Genera N risposte per ogni prompt. Batched inference per efficienza. Output: lista di (response, logprob).

### T4.02 — Rule-Based Math Reward
- **Priorità**: 🔴 Critica | **Stima**: 3d
- **File**: `src/rl/rewards/math_reward.h` (NUOVO)
- **Criteria**: Parser per espressioni math. Eval e confronto con answer attesa. Score: 1.0 corretto, 0.0 sbagliato, parziale per approccio corretto.

### T4.03 — Rule-Based Code Reward
- **Priorità**: 🟠 Alta | **Stima**: 5d
- **File**: `src/rl/rewards/code_reward.h` (NUOVO)
- **Criteria**: Sandbox per compilare ed eseguire codice generato. Confronto output vs expected. Timeout per infinite loop.

### T4.04 — Logic Consistency Reward
- **Priorità**: 🟡 Media | **Stima**: 3d
- **File**: `src/rl/rewards/logic_reward.h` (NUOVO)
- **Criteria**: Check: la risposta contraddice se stessa? La conclusione segue dalle premesse? Heuristic scoring.

### T4.05 — Group Relative Scoring
- **Priorità**: 🔴 Critica | **Stima**: 2d
- **File**: `src/rl/grpo.h` (NUOVO)
- **Criteria**: `r̂_i = (r_i - mean(r)) / std(r)` per N risposte. Normalizzazione per stabilità.

### T4.06 — Policy Gradient con KL Constraint
- **Priorità**: 🔴 Critica | **Stima**: 4d
- **File**: `src/rl/grpo.h` (estendere)
- **Criteria**: `∇J = Σ r̂_i · ∇log π(y_i|x) - β · KL(π || π_ref)`. Aggiornamento pesi con constraint.

### T4.07 — KL Controller Adattivo
- **Priorità**: 🟡 Media | **Stima**: 2d
- **File**: `src/rl/kl_controller.h` (NUOVO)
- **Criteria**: β dinamico: se KL troppo alto → aumenta β, se troppo basso → diminuisci. Target KL configurabile.

---

## 4.2 Thinking Mode

### T4.08 — Thinking Special Tokens
- **Priorità**: 🟠 Alta | **Stima**: 1d
- **File**: `src/tokenizer/`
- **Criteria**: Aggiungere `<thinking>`, `</thinking>` al vocabolario. Token IDs riservati.

### T4.09 — Thinking Router (Trainable)
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/infer/thinking_router.h` (NUOVO)
- **Criteria**: `score = σ(h · W_think)`. Se score > threshold → emit `<thinking>`. W_think trainabile.

### T4.10 — Adaptive Compute Budget
- **Priorità**: 🟡 Media | **Stima**: 3d
- **File**: `src/infer/thinking_router.h` (estendere)
- **Criteria**: Max N thinking tokens configurabile. Early stop se confidence alta. Output: risposta filtrata (thinking rimosso).

### T4.11 — Thinking-Augmented Training Data
- **Priorità**: 🟠 Alta | **Stima**: 5d
- **File**: `src/data/thinking_augment.h` (NUOVO)
- **Criteria**: Genera versioni "thinking" di dataset: domanda → `<thinking>reasoning</thinking>answer`. Usa LLM esterno per generare le tracce di ragionamento.

---

## 4.3 Agent Framework

### T4.12 — Tool Calling Protocol
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/agent/tool_calling.h` (NUOVO)
- **Criteria**: Formato JSON: `{"tool": "name", "args": {...}}`. Parser per output del modello. Dispatch a tool registrato.

### T4.13 — Code Execution Sandbox
- **Priorità**: 🟠 Alta | **Stima**: 5d
- **File**: `src/agent/code_executor.h` (NUOVO)
- **Criteria**: Esegui Python/C++ in processo isolato. Timeout, memory limit. Cattura stdout/stderr.

### T4.14 — Plan-Act-Verify Loop
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/agent/orchestrator.h` (NUOVO)
- **Criteria**: Loop: (1) modello genera piano, (2) esegue azioni, (3) verifica risultato, (4) ripete se necessario.

### T4.15 — Self-Reflection
- **Priorità**: 🟡 Media | **Stima**: 3d
- **File**: `src/agent/reflection.h` (NUOVO)
- **Criteria**: Il modello ri-valuta il proprio output con un prompt di verifica. Score di confidenza.

### T4.16 — Multi-Agent Dispatcher
- **Priorità**: 🟢 Bassa | **Stima**: 5d
- **File**: `src/agent/multi_agent.h` (NUOVO)
- **Criteria**: N agenti specializzati (coder, researcher, reviewer). Orchestratore che assegna task.

---

## 4.4 Evaluation

### T4.17 — MMLU Eval Harness
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/eval/mmlu.h` (NUOVO)
- **Criteria**: 5-shot evaluation su tutte le 57 categories. Accuracy reporting.

### T4.18 — HumanEval Eval
- **Priorità**: 🟠 Alta | **Stima**: 3d
- **File**: `src/eval/humaneval.h` (NUOVO)
- **Criteria**: Genera soluzioni Python, esegui test del benchmark. Pass@1 e Pass@10.

### T4.19 — GSM8K/MATH Eval
- **Priorità**: 🟠 Alta | **Stima**: 2d
- **File**: `src/eval/math_eval.h` (NUOVO)
- **Criteria**: Chain-of-thought prompting. Estrai risposta numerica finale. Confronta con ground truth.

### T4.20 — Custom Multi-Step Reasoning Benchmark
- **Priorità**: 🟡 Media | **Stima**: 4d
- **File**: `src/eval/reasoning_bench.h` (NUOVO)
- **Criteria**: 100 problemi multi-step custom. Valuta correttezza, completezza, coerenza del ragionamento.
