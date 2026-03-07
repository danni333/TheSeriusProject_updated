# Fase 54 — Performance Benchmarks (Throughput, Latenza, Memoria)

> La qualità del modello senza velocità non serve.
> Serve misurare tokens/sec, time-to-first-token, peak VRAM,
> e confrontarsi con llama.cpp e vLLM su hardware identico.

---

## 1. Metriche Chiave

| Metrica | Sigla | Unità | Target (7B Q4, RTX 4090) |
| --- | --- | --- | --- |
| **Tokens per secondo** (decode) | TPS | tok/s | > 80 tok/s |
| **Time To First Token** | TTFT | ms | < 200 ms |
| **Prompt processing** (prefill) | PP | tok/s | > 2000 tok/s |
| **Peak VRAM usage** | VRAM | GB | < 6 GB (7B Q4) |
| **Peak RAM usage** (CPU offload) | RAM | GB | < 8 GB |
| **Batch throughput** | BT | req/s | > 10 req/s (batch=16) |
| **Context window** max | CTX | tokens | 128K senza OOM |
| **Startup time** (model load) | LOAD | sec | < 5 sec (7B Q4) |

---

## 2. Benchmark Suite

### 2.1 Single-User Latency

```cpp
namespace serius::bench {

struct LatencyResult {
    float ttft_ms;           // Time to first token
    float decode_tps;        // Tokens/sec durante generazione
    float prefill_tps;       // Tokens/sec durante prompt processing
    int prompt_tokens;
    int generated_tokens;
    float total_ms;
};

LatencyResult bench_single(Model& model, Tokenizer& tok, Sampler& sampler,
                           const string& prompt, int max_tokens = 256) {
    LatencyResult r;
    auto tokens = tok.encode(prompt);
    r.prompt_tokens = tokens.size();

    // Prefill
    auto t0 = high_resolution_clock::now();
    Tensor logits = model.forward_prefill(tokens);
    auto t1 = high_resolution_clock::now();
    r.ttft_ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0f;
    r.prefill_tps = tokens.size() / (r.ttft_ms / 1000.0f);

    // Decode
    int gen = 0;
    auto t2 = high_resolution_clock::now();
    for (int i = 0; i < max_tokens; i++) {
        int next = sampler.sample(logits);
        if (next == tok.eos_id()) break;
        logits = model.forward_decode(next);
        gen++;
    }
    auto t3 = high_resolution_clock::now();
    float decode_ms = duration_cast<microseconds>(t3 - t2).count() / 1000.0f;
    r.decode_tps = gen / (decode_ms / 1000.0f);
    r.generated_tokens = gen;
    r.total_ms = duration_cast<microseconds>(t3 - t0).count() / 1000.0f;
    return r;
}

} // namespace serius::bench
```

### 2.2 Throughput Batch

```cpp
struct ThroughputResult {
    float requests_per_sec;
    float total_tokens_per_sec;
    float avg_latency_ms;
    float p99_latency_ms;
    int concurrent_users;
};

ThroughputResult bench_throughput(Model& model, int concurrent_users,
                                  int requests_per_user, int max_tokens) {
    // Lancia N thread, ognuno manda requests_per_user richieste
    // Misura throughput aggregato e latenza per-request
    vector<thread> workers;
    atomic<int> total_tokens{0};
    vector<float> latencies;
    mutex lat_mutex;

    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < concurrent_users; i++) {
        workers.emplace_back([&, i]() {
            for (int j = 0; j < requests_per_user; j++) {
                auto result = bench_single(model, tok, sampler, prompts[i], max_tokens);
                total_tokens += result.generated_tokens;
                lock_guard<mutex> lock(lat_mutex);
                latencies.push_back(result.total_ms);
            }
        });
    }
    for (auto& w : workers) w.join();
    auto t1 = high_resolution_clock::now();

    float total_sec = duration_cast<milliseconds>(t1 - t0).count() / 1000.0f;
    sort(latencies.begin(), latencies.end());

    return {
        .requests_per_sec = (concurrent_users * requests_per_user) / total_sec,
        .total_tokens_per_sec = total_tokens / total_sec,
        .avg_latency_ms = accumulate(latencies.begin(), latencies.end(), 0.0f) / latencies.size(),
        .p99_latency_ms = latencies[latencies.size() * 99 / 100],
        .concurrent_users = concurrent_users
    };
}
```

### 2.3 Memory Profiling

```cpp
struct MemoryProfile {
    size_t vram_model_bytes;      // Pesi del modello in VRAM
    size_t vram_kv_cache_bytes;   // KV cache allocata
    size_t vram_activations_bytes; // Attivazioni temporanee
    size_t vram_peak_bytes;       // Picco totale VRAM
    size_t ram_peak_bytes;        // Picco RAM sistema
};

MemoryProfile profile_memory(Model& model, int seq_len, int batch_size) {
    MemoryProfile p;
    // Snapshot prima del loading
    size_t vram_before = hal::get_vram_used();
    model.load("model.gguf");
    p.vram_model_bytes = hal::get_vram_used() - vram_before;

    // Genera con context pieno per misurare KV cache
    vram_before = hal::get_vram_used();
    model.forward_prefill(dummy_tokens(seq_len));
    p.vram_kv_cache_bytes = hal::get_vram_used() - vram_before - p.vram_model_bytes;

    p.vram_peak_bytes = hal::get_vram_peak();
    p.ram_peak_bytes = get_rss_bytes();
    return p;
}
```

---

## 3. Confronto con Competitor

### Tabella comparativa automatica

```bash
# Script che lancia benchmark identici su serius e llama.cpp
serius bench --model model-q4.gguf --prompt-len 512 --gen-tokens 256 \
    --output results_serius.json

llama-bench -m model-q4.gguf -p 512 -n 256 -o json > results_llamacpp.json

python scripts/compare_benchmarks.py \
    --serius results_serius.json \
    --llamacpp results_llamacpp.json \
    --output comparison.md
```

### Output esempio

```text
┌─────────────────┬──────────┬───────────┬──────────┐
│ Metric          │ Serius   │ llama.cpp │ Δ        │
├─────────────────┼──────────┼───────────┼──────────┤
│ Prefill (tok/s) │ 2340     │ 2180      │ +7.3%    │
│ Decode (tok/s)  │ 84.2     │ 81.5      │ +3.3%    │
│ TTFT (ms)       │ 187      │ 205       │ -8.8%    │
│ VRAM (GB)       │ 5.2      │ 5.4       │ -3.7%    │
│ Load time (s)   │ 3.1      │ 3.4       │ -8.8%    │
└─────────────────┴──────────┴───────────┴──────────┘
```

---

## 4. CI Performance Regression

```yaml
# In .github/workflows/perf.yml
perf-regression:
  runs-on: [self-hosted, gpu, linux]
  if: github.event_name == 'pull_request'
  steps:
    - name: Benchmark current PR
      run: ./build/serius bench --model /models/test-1b.gguf --output pr.json
    - name: Compare with baseline
      run: |
        python scripts/check_perf_regression.py \
          --baseline perf/baseline.json --current pr.json \
          --max-regression 5%
```

```python
# scripts/check_perf_regression.py
# Blocca il merge se le performance calano > 5%
for metric in ['decode_tps', 'prefill_tps', 'ttft_ms']:
    baseline_val = baseline[metric]
    current_val = current[metric]
    # Per TPS: regressione se cala. Per TTFT: regressione se cresce.
    if metric.endswith('tps'):
        regression = (baseline_val - current_val) / baseline_val
    else:
        regression = (current_val - baseline_val) / baseline_val
    if regression > threshold:
        fail(f"❌ {metric}: {regression:.1%} regression")
```

---

## 5. Struttura File

| File | Contenuto |
| --- | --- |
| `src/bench/latency.h/.cpp` | Single-user latency benchmark |
| `src/bench/throughput.h/.cpp` | Multi-user throughput benchmark |
| `src/bench/memory.h/.cpp` | Memory profiling (VRAM + RAM) |
| `src/bench/report.h/.cpp` | Report JSON/Markdown/tabella |
| `src/cli/cmd_bench.cpp` | Comando CLI `serius bench` |
| `scripts/compare_benchmarks.py` | Confronto con llama.cpp |
| `scripts/check_perf_regression.py` | CI performance gate |
| `perf/baseline.json` | Baseline performance di riferimento |

---

## 6. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_bench_latency` | TTFT e TPS misurati correttamente |
| `test_bench_memory` | VRAM peak riportato con errore < 5% |
| `test_bench_throughput` | Throughput scala linearmente con batch size |
| `test_perf_regression` | CI blocca merge su regressione > 5% |
| `test_comparison` | Script confronto genera Markdown valido |
