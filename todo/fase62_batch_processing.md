# Fase 62 — Batch Processing Pipeline (Inference Offline su Larga Scala)

> Non tutto è real-time. Serve poter processare migliaia di prompt
> in batch: traduzione massiva, classificazione dataset, generazione dati
> sintetici per training. Ottimizzato per throughput, non latenza.

---

## 1. Casi d'Uso

| Scenario | Input | Output |
| --- | --- | --- |
| **Traduzione corpus** | 10K documenti | 10K traduzioni |
| **Classificazione** | 100K tweet | 100K label + score |
| **Data synthesis** | 1K seed prompt | 50K training examples |
| **Evaluation massiva** | 5K benchmark Q&A | Score per benchmark |
| **Embedding corpus** | 1M documenti | 1M vettori embedding |

---

## 2. CLI

```bash
# Elaborazione batch da file JSONL
serius batch --model model.gguf \
    --input prompts.jsonl \
    --output results.jsonl \
    --max-tokens 256 \
    --concurrent 4 \
    --progress

# Formato input (JSONL)
{"id": "001", "prompt": "Translate to English: Ciao mondo"}
{"id": "002", "prompt": "Translate to English: Buongiorno"}

# Formato output (JSONL)
{"id": "001", "output": "Hello world", "tokens": 3, "time_ms": 45}
{"id": "002", "output": "Good morning", "tokens": 3, "time_ms": 42}
```

---

## 3. Implementazione

```cpp
namespace serius::batch {

struct BatchJob {
    string id;
    string prompt;
    json metadata;
};

struct BatchResult {
    string id;
    string output;
    int tokens_generated;
    float time_ms;
    string error;  // vuoto se ok
};

class BatchProcessor {
    Model& model_;
    int max_concurrent_;
    atomic<int> completed_{0};
    atomic<int> failed_{0};

public:
    void process(const string& input_path, const string& output_path,
                 const SamplerConfig& config) {
        auto jobs = load_jsonl(input_path);
        ofstream out(output_path);

        // Thread pool per processing parallelo
        ThreadPool pool(max_concurrent_);
        for (auto& job : jobs) {
            pool.submit([&, job]() {
                auto result = process_single(job, config);
                write_result(out, result);
                completed_++;
                print_progress(completed_, jobs.size());
            });
        }
        pool.wait_all();
    }
};

} // namespace serius::batch
```

---

## 4. Features Avanzate

- **Resume**: se il processo si interrompe, riparte da dove si era fermato
- **Dedup**: salta prompt duplicati
- **Rate limiting**: limita richieste/sec per non sovraccaricare GPU
- **Error handling**: errori loggati nel JSONL, non bloccano il batch
- **Checkpointing**: salva progresso ogni N risultati

---

## 5. Struttura File

| File | Contenuto |
| --- | --- |
| `src/batch/batch_processor.h/.cpp` | Engine di elaborazione batch |
| `src/batch/jsonl_io.h/.cpp` | Lettura/scrittura JSONL |
| `src/batch/progress.h/.cpp` | Progress bar con ETA |
| `src/cli/cmd_batch.cpp` | Comando `serius batch` |

---

## 6. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_batch_basic` | 100 prompt → 100 risultati JSONL |
| `test_batch_resume` | Interruzione + resume = risultato completo |
| `test_batch_errors` | Prompt malformato → errore nel JSONL, non crash |
| `test_batch_concurrent` | 4 thread paralleli → throughput ~4x |
