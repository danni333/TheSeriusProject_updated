# Fase 23 — RLHF Pipeline (GRPO, DPO, Process Reward Models)

> Dopo SFT (Fase 10), il modello risponde correttamente ma può essere
> "robotico" o sgradevole. RLHF usa un Reward Model addestrato sulle
> preferenze umane per migliorare la qualità percepita.

---

## 1. Pipeline RLHF

```
Fase 1 (già fatta): Pre-training su testo
Fase 10 (già fatta): SFT su coppie istruzione-risposta
     │
     ▼
Fase 23a: Addestra Reward Model
     │    (su coppie "risposta A vs B, quale preferisci?")
     │
     ▼
Fase 23b: Ottimizza il modello con GRPO (Group Relative Policy Optimization)
          (Nativamente implementato alla DeepSeek per ridurre la memoria)
```

---

## 2. Reward Model

### Architettura

Stesso Transformer del modello principale, ma con una **testa scalare** al posto della testa di vocabolario:

```
Input: [prompt + risposta]
     │
     ▼
Transformer (può essere il modello SFT stesso)
     │
     ▼
Ultimo hidden state → Linear(d_model, 1) → scalare (reward score)
```

### Training del Reward Model

Dataset: coppie di risposte con preferenza umana.

```
Prompt: "Cos'è la fotosintesi?"
Risposta A: "È il processo con cui le piante..." (preferita ✓)
Risposta B: "Le piante mangiano la luce..."      (peggiore ✗)
```

### Loss: Bradley-Terry

$$L = -\log\sigma(r(x, y_w) - r(x, y_l))$$

dove $y_w$ = risposta preferita, $y_l$ = risposta peggiore.

```cpp
float reward_loss(float score_win, float score_lose) {
    return -logf(sigmoid(score_win - score_lose));
}

// Training loop
for (auto& [prompt, resp_win, resp_lose] : dataset) {
    float r_win  = reward_model.score(prompt + resp_win);
    float r_lose = reward_model.score(prompt + resp_lose);
    float loss = reward_loss(r_win, r_lose);
    loss.backward();
    optimizer.step();
}
```

---

## 3. GRPO (Group Relative Policy Optimization)

### Concetto

GRPO è l'alternativa iper-efficiente a PPO usata da DeepSeek (V2/V3/R1). PPO richiede un **Value Model** separato (delle stesse dimensioni del Policy) che prevede il reward per calcolare l'Advantage. Questo raddoppia la memoria richiesta.  
**GRPO elimina il Value Model** campionando $G$ output diversi per lo stesso prompt e calcolando l'Advantage in modo relativo (rispetto alla media del gruppo).

### Algoritmo semplificato GRPO

```
Per ogni batch di prompt:
    1. Genera G risposte per ogni prompt con il modello (π_old)
    2. Calcola i reward R_1, ..., R_G usando il Reward Model (o Rule-based verifier)
    3. Calcola l'Advantage relativo: A_i = (R_i - mean(R)) / std(R)
    4. Aggiorna πθ massimizzando l'obiettivo:

Objective:
    ratio_i = π_new(a_i|s) / π_old(a_i|s)
    L = Σ_i [ min(ratio_i · A_i, clip(ratio_i, 1-ε, 1+ε) · A_i) - β · D_KL(π_new || π_ref) ]
```

### Implementazione

```cpp
struct PPOTrainer {
    Model& policy;           // modello da ottimizzare
    Model& ref_policy;       // copia frozen (per KL penalty)
    RewardModel& reward;
    AdamW optimizer;
    float clip_eps = 0.2f;
    float kl_coeff = 0.1f;

    void train_step(const vector<string>& prompts) {
        for (auto& prompt : prompts) {
            // 1. Genera con policy corrente
            auto [tokens, log_probs_old] = policy.generate_with_logprobs(prompt);
            string response = tokenizer.decode(tokens);

            // 2. Reward
            float reward_score = reward.score(prompt + response);

            // 3. KL penalty (evita che il modello diverga troppo)
            auto log_probs_ref = ref_policy.compute_logprobs(prompt + response);
            float kl = mean(log_probs_old - log_probs_ref);
            float final_reward = reward_score - kl_coeff * kl;

            // 4. Advantage (simplified: reward - baseline)
            float advantage = final_reward - baseline;

            // 5. PPO update
            auto log_probs_new = policy.compute_logprobs(prompt + response);
            for (int t = 0; t < tokens.size(); t++) {
                float ratio = expf(log_probs_new[t] - log_probs_old[t]);
                float clipped = clamp(ratio, 1-clip_eps, 1+clip_eps);
                float loss = -fmin(ratio * advantage, clipped * advantage);
                // accumulate loss, backward, step
            }
        }
    }
};
```

### Componenti in memoria

```
### Componenti in memoria (PPO vs GRPO)

```

        PPO MEMORY                       GRPO MEMORY
┌────────────────────────┐      ┌────────────────────────┐
│ Policy Model           │      │ Policy Model           │
│ Reference Model        │      │ Reference Model        │
│ Reward Model           │      │ Reward Model           │
│ Value Model (CRITICO!) │      │                        │ ← NESSUN VALUE MODEL!
│ Optimizer states (m,v) │      │ Optimizer states (m,v) │
└────────────────────────┘      └────────────────────────┘

```

> [!CAUTION]
> Durante RLHF servono moltissima memoria. GRPO rimuove uno dei componenti più onerosi, permettendo di addestrare un 7B su singole macchine con memorie limitate o ottimizzando pesantemente il training su cluster multi-nodo.
```

> [!CAUTION]
> RLHF richiede **3 modelli in memoria** simultaneamente.
> Servono almeno 2 GPU da 40+ GB o tecniche di offloading aggressivo.

---

## 4. KL Divergence Penalty

Evita che il modello "collassi" generando solo frasi che massimizzano il reward:

$$R_{final} = R_{reward} - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

```cpp
float kl_divergence(const vector<float>& log_p, const vector<float>& log_q) {
    float kl = 0;
    for (size_t i = 0; i < log_p.size(); i++)
        kl += expf(log_p[i]) * (log_p[i] - log_q[i]);
    return kl;
}
```

---

## 5. Struttura File

| File | Contenuto |
|---|---|
| `src/rlhf/reward_model.h/.cpp` | Reward Model (Transformer + scalar head) |
| `src/rlhf/ppo.h/.cpp` | PPO trainer (clipped objective, GAE) |
| `src/rlhf/kl_penalty.h/.cpp` | KL divergence calcolo |
| `src/rlhf/rollout.h/.cpp` | Generazione con log-probs |
| `tests/test_reward.cpp` | Preferenza A>B → score_A > score_B |
| `tests/test_ppo.cpp` | PPO step riduce loss su toy problem |

---

## 6. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_reward_ranking` | Risposta buona ha score > risposta cattiva |
| `test_reward_loss` | Bradley-Terry loss decresce durante training |
| `test_ppo_clip` | Ratio clipped a [1-ε, 1+ε] |
| `test_kl_penalty` | KL > 0 quando policy ≠ ref, KL = 0 quando uguali |
| `test_ppo_improvement` | Dopo 10 step PPO, reward medio aumenta |
| `test_no_collapse` | Con KL penalty, output rimane diversificato |
