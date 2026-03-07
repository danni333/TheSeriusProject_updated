# Fase 28 — Monitoring, Logging e Telemetria

> In produzione devi sapere cosa succede "dentro" il modello:
> latenza, VRAM, throughput, e Continuous Batching per massimizzare la GPU.

---

## 1. Prometheus Metrics

### Metriche esposte

```cpp
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>

class InferenceMetrics {
    prometheus::Counter& requests_total;
    prometheus::Counter& tokens_generated;
    prometheus::Gauge& active_requests;
    prometheus::Gauge& vram_usage_bytes;
    prometheus::Gauge& gpu_temperature;
    prometheus::Histogram& ttft_seconds;        // Time To First Token
    prometheus::Histogram& tps;                 // Tokens Per Second
    prometheus::Histogram& request_duration;

    void on_request_start() {
        requests_total.Increment();
        active_requests.Increment();
    }
    void on_token_generated(float latency_ms) {
        tokens_generated.Increment();
        tps.Observe(1000.0f / latency_ms);
    }
    void on_request_end(float duration_s) {
        active_requests.Decrement();
        request_duration.Observe(duration_s);
    }
};
```

### Endpoint `/metrics`

```cpp
// Espone metriche in formato Prometheus
app.route("/metrics", [&](auto& req) {
    return prometheus::TextSerializer().Serialize(registry.Collect());
});
```

```
# Output esempio
llm_requests_total 1523
llm_tokens_generated_total 45690
llm_active_requests 3
llm_vram_usage_bytes 14523867136
llm_gpu_temperature_celsius 72
llm_ttft_seconds_bucket{le="0.1"} 1200
llm_tps_bucket{le="30"} 1450
```

---

## 2. Grafana Dashboard

### Pannelli consigliati

| Pannello | Query PromQL |
|---|---|
| Request rate | `rate(llm_requests_total[5m])` |
| Token throughput | `rate(llm_tokens_generated_total[1m])` |
| TTFT P99 | `histogram_quantile(0.99, llm_ttft_seconds_bucket)` |
| VRAM usage | `llm_vram_usage_bytes / 1024^3` (in GB) |
| GPU temp | `llm_gpu_temperature_celsius` |
| Active requests | `llm_active_requests` |
| Error rate | `rate(llm_errors_total[5m])` |

---

## 3. Continuous Batching

### Problema
Servire un utente alla volta spreca GPU:
```
User A: ████████████████░░░░░░░░  (GPU 60% idle)
User B:               ████████████████  (aspetta A)
```

### Soluzione: Continuous Batching

```
Iteration 1: [A_tok1, B_tok1, C_tok1]  → 3 utenti in parallelo
Iteration 2: [A_tok2, B_tok2, C_tok2]
Iteration 3: [A_tok3, B_EOS,  C_tok3]  → B finisce, slot libero
Iteration 4: [A_tok4, D_tok1, C_tok4]  → D entra nello slot di B
```

### Implementazione

```cpp
class ContinuousBatcher {
    struct Slot {
        int request_id;
        vector<int> tokens;
        KVCacheSlice kv;
        function<void(string, bool)> callback;
        bool active = false;
    };

    vector<Slot> slots;       // max_batch_size slots
    queue<Request> waiting;

    void iteration() {
        // 1. Riempi gli slot vuoti con nuove richieste
        for (auto& slot : slots)
            if (!slot.active && !waiting.empty())
                fill_slot(slot, waiting.front()), waiting.pop();

        // 2. Batch forward: tutti gli slot attivi insieme
        vector<int> active_indices;
        for (int i = 0; i < slots.size(); i++)
            if (slots[i].active) active_indices.push_back(i);

        Tensor batch_logits = model.forward_batch(slots, active_indices);

        // 3. Sample e dispatch per ogni slot
        for (int i : active_indices) {
            int next_token = sampler.sample(batch_logits[i]);
            slots[i].tokens.push_back(next_token);

            string text = tokenizer.decode({next_token});
            bool done = (next_token == EOS || slots[i].tokens.size() >= max_len);
            slots[i].callback(text, done);

            if (done) slots[i].active = false;  // libera slot
        }
    }
};
```

---

## 4. Structured Logging

```cpp
// JSON logging per analisi e debug
void log_request(const Request& req, const Response& resp, float duration_ms) {
    json log = {
        {"timestamp", now_iso8601()},
        {"request_id", req.id},
        {"model", model_name},
        {"prompt_tokens", req.prompt_tokens},
        {"completion_tokens", resp.tokens.size()},
        {"ttft_ms", resp.time_to_first_token_ms},
        {"total_ms", duration_ms},
        {"tokens_per_sec", resp.tokens.size() / (duration_ms / 1000)},
        {"finish_reason", resp.finish_reason}
    };
    logger.info(log.dump());
}
```

---

## 5. Struttura File

| File | Contenuto |
|---|---|
| `src/server/metrics.h/.cpp` | Prometheus metrics |
| `src/server/batcher.h/.cpp` | Continuous Batching engine |
| `src/server/logging.h/.cpp` | Structured JSON logging |
| `grafana/dashboard.json` | Dashboard Grafana pre-configurata |
| `tests/test_batcher.cpp` | Continuous batching: slot fill, EOS |

---

## 6. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_metrics_endpoint` | `/metrics` restituisce formato Prometheus valido |
| `test_continuous_batch` | 4 richieste servite in parallelo, ognuna riceve i propri token |
| `test_slot_reuse` | Slot liberato dopo EOS, riusato per nuova richiesta |
| `test_logging` | Log JSON valido con tutti i campi |
