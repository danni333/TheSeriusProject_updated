# Fase 25 — Benchmarking Autonomo (MMLU-Pro, IFEval, LiveCodeBench, ELO Arena)

> Come fai a sapere se la v2.0 è meglio della v1.0?
> Servono test automatici e analisi della convergenza.

---

## 1. Evals Suite

### Benchmark standard

| Benchmark | Cosa misura | Formato |
|---|---|---|
| **MMLU** | Conoscenza generale (57 materie) | Multiple choice (A/B/C/D) |
| **GSM8K** | Ragionamento matematico | Problema → soluzione numerica |
| **HumanEval** | Generazione codice Python | Funzione → unit test pass/fail |
| **HellaSwag** | Ragionamento di buon senso | Completamento frase |
| **TruthfulQA** | Veridicità delle risposte | Multiple choice + generazione |

### Implementazione

```cpp
struct EvalResult {
    string benchmark;
    int correct, total;
    float accuracy() const { return (float)correct / total; }
};

class EvalSuite {
    Model& model;
    Sampler& sampler;

    // MMLU: multiple choice
    EvalResult eval_mmlu(const string& dataset_path) {
        auto questions = load_mmlu(dataset_path);
        int correct = 0;
        for (auto& q : questions) {
            string prompt = format_mmlu_prompt(q);
            // Genera un singolo token (A/B/C/D)
            Tensor logits = model.forward(tokenizer.encode(prompt));
            int pred = argmax_over({token_A, token_B, token_C, token_D}, logits);
            if (pred == q.answer) correct++;
        }
        return {"MMLU", correct, (int)questions.size()};
    }

    // GSM8K: chain-of-thought → numero finale
    EvalResult eval_gsm8k(const string& dataset_path) {
        auto problems = load_gsm8k(dataset_path);
        int correct = 0;
        for (auto& p : problems) {
            string response = model.generate(p.question, {.max_tokens=512});
            int answer = extract_final_number(response);
            if (answer == p.expected) correct++;
        }
        return {"GSM8K", correct, (int)problems.size()};
    }

    // Report completo
    void run_all(const string& output_path) {
        vector<EvalResult> results;
        results.push_back(eval_mmlu("data/mmlu/"));
        results.push_back(eval_gsm8k("data/gsm8k/"));
        // ... altri benchmark
        save_report(results, output_path);
    }
};
```

---

## 2. Perplexity

Misura fondamentale: quanto il modello è "sorpreso" dal testo.

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log p(x_i | x_{<i})\right)$$

```cpp
float compute_perplexity(Model& model, const vector<int>& tokens) {
    float total_log_prob = 0;
    for (int i = 1; i < tokens.size(); i++) {
        Tensor logits = model.forward(tokens, i);
        Tensor probs = softmax(logits);
        total_log_prob += logf(probs[tokens[i]] + 1e-10f);
    }
    return expf(-total_log_prob / (tokens.size() - 1));
}
```

---

## 3. Loss Curve Analysis

### Overfitting Detection

```cpp
struct TrainingMetrics {
    vector<float> train_loss;
    vector<float> val_loss;
    vector<int> steps;

    bool is_overfitting(int window = 100) {
        if (val_loss.size() < window) return false;
        // Overfitting: train loss scende, val loss sale
        float train_trend = trend(train_loss, window);  // negativo = scende
        float val_trend = trend(val_loss, window);      // positivo = sale
        return train_trend < -0.01f && val_trend > 0.01f;
    }

    void export_csv(const string& path) {
        ofstream f(path);
        f << "step,train_loss,val_loss\n";
        for (int i = 0; i < steps.size(); i++)
            f << steps[i] << "," << train_loss[i]
              << "," << val_loss[i] << "\n";
    }
};
```

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `src/eval/eval_suite.h/.cpp` | Runner per MMLU, GSM8K, HumanEval |
| `src/eval/perplexity.h/.cpp` | Calcolo perplexity |
| `src/eval/metrics.h/.cpp` | Loss curve analysis, overfitting detection |
| `src/eval/report.h/.cpp` | Esportazione risultati (JSON, CSV) |
| `tests/test_eval.cpp` | Test correttezza evals su dati sintetici |

---

## 5. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_mmlu_format` | Prompt formattato correttamente |
| `test_perplexity` | PPL su testo uniforme = vocab_size |
| `test_overfitting` | Detection scatta quando val_loss sale |
| `test_report` | CSV/JSON valido e leggibile |
