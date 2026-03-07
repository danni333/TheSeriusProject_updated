# Fase 59 — A/B Testing e Model Comparison

> Come sai se la v2 è davvero meglio della v1?
> Framework per confrontare modelli side-by-side con
> blind evaluation, ELO rating automatico, e split traffic.

---

## 1. Modalità di Confronto

### 1.1 Blind A/B Test (Arena-style)

```text
Utente manda prompt → entrambi i modelli rispondono → utente sceglie il migliore

┌─────────────────────────────────────────┐
│ Prompt: "Spiega la relatività generale" │
├────────────────────┬────────────────────┤
│ Modello A (hidden) │ Modello B (hidden) │
│ "La relatività..." │ "Einstein nel..."  │
├────────────────────┴────────────────────┤
│ 👈 Preferisco A  │  Pari  │ Preferisco B 👉 │
└─────────────────────────────────────────┘
```

### 1.2 ELO Rating Automatico

```cpp
namespace serius::eval {

struct ELOSystem {
    map<string, float> ratings;  // model_id → ELO score
    float k_factor = 32.0f;

    void update(const string& winner, const string& loser) {
        float Ea = expected(ratings[winner], ratings[loser]);
        float Eb = 1.0f - Ea;
        ratings[winner] += k_factor * (1.0f - Ea);
        ratings[loser]  += k_factor * (0.0f - Eb);
    }

    float expected(float Ra, float Rb) {
        return 1.0f / (1.0f + powf(10.0f, (Rb - Ra) / 400.0f));
    }
};

} // namespace serius::eval
```

### 1.3 Split Traffic in Produzione

```json
// config/ab_test.json
{
    "experiment": "v2_vs_v1",
    "models": {
        "control": { "path": "models/v1.gguf", "traffic": 50 },
        "variant": { "path": "models/v2.gguf", "traffic": 50 }
    },
    "metrics": ["user_preference", "response_time", "token_count"],
    "duration_hours": 168
}
```

---

## 2. CLI

```bash
# Arena locale
serius arena --model-a v1.gguf --model-b v2.gguf --rounds 50

# Report ELO
serius arena --results arena_log.jsonl --report
# Output:
# 🏆 ELO Rankings:
#   1. v2.gguf     — 1087 (±24)
#   2. v1.gguf     — 913  (±24)
#   Matches: 50 | v2 wins: 32 | v1 wins: 14 | Ties: 4
```

---

## 3. Struttura File

| File | Contenuto |
| --- | --- |
| `src/eval/arena.h/.cpp` | Blind A/B comparison engine |
| `src/eval/elo.h/.cpp` | ELO rating system |
| `src/eval/ab_splitter.h/.cpp` | Traffic splitting per experiments |
| `src/cli/cmd_arena.cpp` | Comando `serius arena` |

---

## 4. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_elo_convergence` | ELO converge dopo 100+ match |
| `test_blind_mode` | Nomi modelli nascosti all'utente |
| `test_traffic_split` | 50/50 split effettivo (±5%) |
| `test_arena_report` | Report Markdown/JSON valido |
