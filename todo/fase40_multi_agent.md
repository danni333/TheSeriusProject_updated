# Fase 40 — Multi-Agent Orchestration (AutoGen-style, Cognitive Architecture)

> L'ultimo gradino: il modello non lavora da solo,
> ma coordina altri modelli come un "cervello centrale".

---

## 1. Supervisor Pattern

### Architettura

```
                    ┌─────────────────┐
                    │   SUPERVISOR    │
    User query ───▶│  (Core LLM 7B)  │───▶ Final answer
                    │  Capisce, delega│
                    └───┬───┬───┬────┘
                        │   │   │
            ┌───────────┼───┼───┼───────────┐
            ▼           ▼   │   ▼           ▼
    ┌──────────┐ ┌─────────┐│┌─────────┐ ┌──────────┐
    │ SQL Agent│ │Translate ││ │WebSearch│ │Code Agent│
    │  (125M)  │ │  (125M)  ││ │  (API)  │ │  (1.5B)  │
    └──────────┘ └─────────┘│└─────────┘ └──────────┘
                            │
                     ┌──────────┐
                     │Math Agent│
                     │  (500M)  │
                     └──────────┘
```

### Implementazione

```cpp
class Supervisor {
    Model& core;
    map<string, Agent*> specialists;

    string process(const string& query) {
        // 1. Il core analizza la query
        string plan = core.generate(
            "Analizza questa richiesta e decidi quali specialisti usare:\n"
            + query + "\nSpecialisti: " + list_specialists());

        // 2. Parsa il piano ed esegui i task
        auto tasks = parse_plan(plan);
        map<string, string> results;
        for (auto& task : tasks) {
            Agent* agent = specialists[task.agent_name];
            results[task.id] = agent->execute(task.prompt);
        }

        // 3. Il core sintetizza la risposta
        return core.generate(format_synthesis(query, results));
    }
};

struct Agent {
    string name;
    string specialty;
    Model* model;       // modello specializzato (può essere piccolo)
    MCPClient* tools;   // tool MCP opzionali

    string execute(const string& task) {
        if (tools) {
            // Agent con tool (ReAct)
            ReActEngine react(*model, *tools);
            return react.run(task);
        }
        return model->generate(task);
    }
};
```

---

## 2. Consensus Algorithms (Metodo Delphi)

### Concetto

Più istanze del modello "dibattono" per risolvere un problema complesso.

```
Round 1: 3 agenti generano risposte indipendenti
Round 2: Ogni agente vede le risposte degli altri e rivede la propria
Round 3: Se c'è consenso → risposta finale
         Se no → voto di maggioranza
```

```cpp
class DelphiConsensus {
    vector<Model*> agents;
    int max_rounds = 3;

    string debate(const string& question) {
        vector<string> responses(agents.size());

        for (int round = 0; round < max_rounds; round++) {
            for (int i = 0; i < agents.size(); i++) {
                string prompt = question;
                if (round > 0) {
                    prompt += "\n\nRisposte degli altri partecipanti:\n";
                    for (int j = 0; j < agents.size(); j++)
                        if (j != i) prompt += format("Agent {}: {}\n", j, responses[j]);
                    prompt += "\nRiconsidera la tua risposta:";
                }
                responses[i] = agents[i]->generate(prompt, {.temp=0.7f});
            }
            // Controlla consenso
            if (check_consensus(responses))
                return responses[0];
        }
        // Fallback: voto di maggioranza
        return majority_vote(responses);
    }

    bool check_consensus(const vector<string>& responses) {
        for (int i = 1; i < responses.size(); i++)
            if (cosine_sim(embed(responses[0]), embed(responses[i])) < 0.9f)
                return false;
        return true;
    }
};
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/agent/supervisor.h/.cpp` | Supervisor pattern con delegation |
| `src/agent/specialist.h/.cpp` | Agenti specializzati |
| `src/agent/consensus.h/.cpp` | Delphi consensus algorithm |
| `src/agent/planner.h/.cpp` | Task decomposition e planning |
| `tests/test_multi_agent.cpp` | Supervisor delega, consensus converge |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_delegation` | Supervisor delega "traduci" al Translation Agent |
| `test_synthesis` | Risultati multipli sintetizzati in risposta coerente |
| `test_consensus_converge` | 3 agenti convergono in ≤ 3 round |
| `test_majority_vote` | 2 su 3 d'accordo → risposta maggioritaria |
| `test_agent_isolation` | Errore di un agente non blocca il supervisor |
