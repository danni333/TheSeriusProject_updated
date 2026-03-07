# Fase 61 — Skill System (Creazione, Condivisione e Uso di Skill)

> Un tool esegue **una singola azione**. Una skill è un **workflow completo**:
> una sequenza di istruzioni, prompt, tool e logica che il modello può
> apprendere, eseguire e condividere. Come le app per uno smartphone.

---

## 1. Cosa sono le Skill

```text
Skill = System Prompt + Tool Requirements + Workflow Steps + Examples

Esempio: "Analisi Codebase"
  ├── System Prompt: "Sei un senior developer che analizza codice..."
  ├── Tools richiesti: [filesystem, grep, code_analyzer]
  ├── Steps:
  │   1. Lista tutti i file nel progetto
  │   2. Identifica linguaggio principale
  │   3. Analizza dipendenze
  │   4. Genera report architetturale
  └── Output: report.md
```

---

## 2. Formato Skill (YAML)

```yaml
# skills/code_review.yaml
name: "Code Review"
version: "1.0.0"
author: "community"
description: "Revisione automatica del codice con best practices"
license: "MIT"

# Requisiti
requires:
  tools: ["filesystem", "grep"]
  min_context: 8192
  min_model_size: "3B"

# System prompt della skill
system_prompt: |
  Sei un code reviewer esperto. Analizza il codice fornito e produci
  un report con: bug potenziali, code smells, suggerimenti di refactoring,
  e problemi di sicurezza. Usa le best practices del linguaggio.

# Parametri configurabili dall'utente
parameters:
  - name: "language"
    type: "string"
    default: "auto"
    description: "Linguaggio di programmazione (auto-detect se vuoto)"
  - name: "severity"
    type: "enum"
    values: ["low", "medium", "high"]
    default: "medium"
    description: "Soglia minima di severità per i finding"

# Workflow (sequenza di step)
workflow:
  - step: "read_files"
    action: "tool:filesystem.list"
    args: { path: "{{input_path}}", recursive: true }

  - step: "analyze"
    action: "llm:generate"
    prompt: |
      Analizza i seguenti file e trova problemi:
      {{read_files.output}}
    max_tokens: 2048

  - step: "format_report"
    action: "llm:generate"
    prompt: |
      Formatta i risultati come report Markdown:
      {{analyze.output}}

# Esempi per few-shot
examples:
  - input: "Analizza src/main.cpp"
    output: "## Code Review Report\n### Bug:\n- Line 42: possibile null dereference..."
```

---

## 3. Skill Manager

```cpp
namespace serius::skill {

struct Skill {
    string name, version, author, description;
    string system_prompt;
    vector<string> required_tools;
    vector<WorkflowStep> workflow;
    vector<Parameter> parameters;
};

class SkillManager {
public:
    // Carica skill da file YAML
    Skill load(const string& path);

    // Carica tutte le skill da directory
    vector<Skill> load_all(const string& skills_dir);

    // Esegui una skill
    string execute(const Skill& skill, const json& user_input,
                   Model& model, ToolRouter& tools);

    // Installa skill da registry online
    void install(const string& skill_id);

    // Lista skill disponibili
    vector<SkillInfo> list_available();

    // Crea nuova skill interattivamente
    Skill create_interactive(Model& model);
};

class SkillExecutor {
    Model& model_;
    ToolRouter& tools_;

public:
    // Esegue ogni step del workflow in sequenza
    json run_workflow(const Skill& skill, const json& params);

private:
    json run_step(const WorkflowStep& step, const json& context);
    json run_tool_step(const string& tool_action, const json& args);
    json run_llm_step(const string& prompt, int max_tokens);
};

} // namespace serius::skill
```

---

## 4. CLI

```bash
# Lista skill installate
serius skill list
# Output:
# ├── code_review     v1.0.0  "Code review automatico"
# ├── data_analysis   v2.1.0  "Analisi dataset CSV/JSON"
# ├── web_research    v1.3.0  "Ricerca e sintesi dal web"
# └── translate_doc   v1.0.0  "Traduzione documenti"

# Esegui skill
serius skill run code_review --path ./src/ --severity high

# Installa skill da registry
serius skill install community/summarize-paper

# Crea nuova skill (guidato dal modello)
serius skill create
# > Nome: "email_writer"
# > Descrizione: "Scrivi email professionali"
# > Tool necessari: nessuno
# > ✅ Skill salvata in skills/email_writer.yaml

# Condividi skill
serius skill publish email_writer
```

---

## 5. Skill Store / Registry

```text
serius.dev/skills/
├── official/              ← Skill ufficiali mantenute dal team
│   ├── code_review
│   ├── data_analysis
│   └── summarize
│
├── community/             ← Skill della community
│   ├── translate_doc
│   ├── web_research
│   └── email_writer
│
└── local/                 ← Skill create dall'utente
    └── my_custom_skill
```

---

## 6. Struttura File

| File | Contenuto |
| --- | --- |
| `src/skill/skill.h` | Struct Skill, Parameter, WorkflowStep |
| `src/skill/skill_manager.h/.cpp` | Load, install, list, create |
| `src/skill/skill_executor.h/.cpp` | Esecuzione workflow step-by-step |
| `src/skill/skill_registry.h/.cpp` | Download/upload da registry online |
| `src/cli/cmd_skill.cpp` | Comandi `serius skill *` |
| `skills/` | Directory skill installate |
| `skills/code_review.yaml` | Esempio skill built-in |
| `docs/skill-development.md` | Guida per creare skill custom |

---

## 7. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_skill_load` | YAML parsato correttamente in struct Skill |
| `test_skill_execute` | Workflow multi-step produce output sensato |
| `test_skill_params` | Parametri utente sostituiti nel template |
| `test_skill_tools` | Step con tool chiama tool router correttamente |
| `test_skill_create` | Creazione interattiva genera YAML valido |
| `test_skill_install` | Install da registry scarica e carica skill |
