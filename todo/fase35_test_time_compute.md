# Fase 35 — Test-Time Compute & System 2 Reasoning (o1/R1 Style, PRM Verifiers)

> Il modello non deve dare la prima risposta che gli viene in mente ("System 1").
> L'allocazione di computazione durante l'inferenza (Test-Time Compute) permette a un modello di correggere i propri errori, deliberare e pianificare ("System 2").

---

## 1. Multi-Step Chain of Thought Formattata (R1 / o1 Style)

### Buffer di pensiero separato (`<think>`)

```cpp
struct ThinkingResult {
    string thinking;    // buffer nascosto (non mostrato all'utente)
    string answer;      // risposta finale
};

ThinkingResult generate_with_thinking(Model& model, const string& prompt) {
    string full_prompt = prompt + "\n<think>\n";
    
    // Genera pensiero in buffer separato
    string thinking = model.generate(full_prompt, {
        .stop_tokens = {"</think>"},
        .max_tokens = 2048
    });
    
    // Genera risposta basata sul pensiero
    string answer_prompt = full_prompt + thinking + "</think>\n<answer>\n";
    string answer = model.generate(answer_prompt, {
        .stop_tokens = {"</answer>"},
        .max_tokens = 1024
    });
    
    return {thinking, answer};
}
```

### Compute budget adattivo

```cpp
int estimate_thinking_budget(const string& query) {
    // Query semplice: pochi token di pensiero
    // Query complessa (math, coding): molti token
    if (contains_math(query))  return 4096;
    if (contains_code(query))  return 3072;
    if (is_factual(query))     return 512;
    return 1024;  // default
}
```

---

## 2. Reinforcement Learning-based Search e MCTS

### Concetto

Mentre il normale RLHF ottimizza il risultato finale ("Outcome Reward"), il ragionamento a Test-Time ha bisogno di validatori intermedi, noti come **PRM (Process Reward Models)**, o di esplorare tramite **RL Search procedures** (come implementato in GPT-o1 o DeepSeek-R1-Zero).

Esplorare diverse strade di ragionamento test-time come un albero:

```
                    [Problema]
                   /    |     \
            [Step A1] [Step B1] [Step C1]
            /    \       |
     [Step A2a] [A2b]  [B2]
        |
    [Answer]  ← miglior percorso
```

### Algoritmo

```cpp
struct MCTSNode {
    string text;           // testo generato fino a qui
    float value = 0;       // score medio dei rollout
    int visits = 0;
    vector<MCTSNode*> children;
    MCTSNode* parent = nullptr;

    float ucb1(float c = 1.414f) {
        if (visits == 0) return INFINITY;
        return value / visits + c * sqrtf(logf(parent->visits) / visits);
    }
};

class MCTSReasoner {
    Model& model;
    Model& verifier;  // modello che valuta la correttezza
    int simulations;

    string search(const string& problem) {
        MCTSNode root = {problem};
        
        for (int sim = 0; sim < simulations; sim++) {
            // 1. SELECT: scegli il nodo con UCB1 più alto
            MCTSNode* node = select(&root);
            
            // 2. EXPAND: genera N possibili step successivi
            expand(node);
            
            // 3. EVALUATE: valuta la qualità del ragionamento
            float score = evaluate(node);
            
            // 4. BACKPROPAGATE: aggiorna score risalendo
            backpropagate(node, score);
        }
        
        // Restituisci il percorso migliore
        return best_path(&root);
    }

    void expand(MCTSNode* node) {
        // Genera 3-5 possibili continuazioni
        for (int i = 0; i < 5; i++) {
            string next = model.generate(node->text + "\nStep: ",
                {.temp=0.8+i*0.1, .max_tokens=256, .stop_tokens={"\n"}});
            node->children.push_back(new MCTSNode{node->text + next, 0, 0, {}, node});
        }
    }

    float evaluate(MCTSNode* node) {
        // Rollout: completa il ragionamento e valuta
        string completion = model.generate(node->text + "\nAnswer: ", {.max_tokens=512});
        return verifier.score(node->text + completion);  // 0-1
    }
};
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/reasoning/cot.h/.cpp` | Chain of Thought con buffer separato |
| `src/reasoning/mcts.h/.cpp` | Monte Carlo Tree Search |
| `src/reasoning/verifier.h/.cpp` | Modello verificatore di ragionamento |
| `src/reasoning/budget.h/.cpp` | Compute budget adattivo |
| `tests/test_reasoning.cpp` | CoT produce pensiero + risposta, MCTS trova soluzione |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_cot_separation` | Thinking e answer in buffer separati |
| `test_cot_improves` | Con CoT: accuracy su GSM8K > senza CoT |
| `test_mcts_explores` | Almeno 3 percorsi diversi esplorati |
| `test_mcts_finds_best` | Percorso con score più alto selezionato |
| `test_budget_adaptive` | Query matematica → budget alto |
