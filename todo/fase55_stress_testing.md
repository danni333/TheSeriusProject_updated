# Fase 55 — Stress Testing & Load Testing

> Un server che funziona con 1 utente non è pronto per la produzione.
> Serve testare i limiti: massimo throughput, stabilità sotto carico,
> leak di memoria, e recovery dopo errori.

---

## 1. Scenari di Test

### 1.1 Ramping Load Test

```text
Scenario: aumenta gradualmente il carico fino al punto di rottura

Utenti concorrenti:  1 → 2 → 4 → 8 → 16 → 32 → 64 → 128
Durata per step:     60 secondi
Prompt:              512 token (fixed)
Max tokens:          256

Metriche tracciate:
  - TPS aggregato (deve salire, poi plateau)
  - P99 latency (deve restare < 10s)
  - Error rate (deve restare 0% fino a saturazione)
  - VRAM usage (non deve superare il limite fisico)
```

### 1.2 Soak Test (durata prolungata)

```text
Scenario: carico costante per ore, cerca memory leak

Durata:              4 ore
Utenti concorrenti:  8 (costanti)
Richieste:           1 ogni 5 secondi per utente 

Verifica:
  - VRAM non cresce nel tempo (leak detection)
  - RAM non cresce nel tempo
  - TPS stabile (no degradazione)
  - Zero crash / zero riavvii
```

### 1.3 Spike Test

```text
Scenario: picco improvviso di traffico

Fase 1 (normal):     4 utenti per 30 sec
Fase 2 (spike):      64 utenti per 60 sec
Fase 3 (recovery):   4 utenti per 30 sec

Verifica:
  - Il server non crasha durante lo spike
  - Le richieste in eccesso ricevono 429 (non timeout)
  - Recovery a TPS normale dopo lo spike
```

### 1.4 Chaos Test

```text
Scenario: errori casuali durante il funzionamento

Test:
  - Input vuoto → risposta di errore, non crash
  - Context length > max → errore graceful
  - Token malformati → errore 400, non segfault
  - Disconnessione client durante streaming → cleanup KV cache
  - OOM simulato → error handling, non abort()
```

---

## 2. Implementazione

```cpp
namespace serius::stress {

struct StressConfig {
    int concurrent_users;
    int duration_seconds;
    int ramp_step_seconds = 60;
    string prompt = "Explain quantum computing in detail...";
    int max_tokens = 256;
    string server_url = "http://localhost:8080";
};

struct StressResult {
    float avg_tps;
    float p50_latency_ms;
    float p99_latency_ms;
    float error_rate;
    size_t total_requests;
    size_t total_errors;
    vector<pair<float, float>> tps_timeline;  // (timestamp, tps)
    vector<pair<float, size_t>> vram_timeline; // (timestamp, bytes)
};

class StressRunner {
public:
    StressResult run_ramp(const StressConfig& cfg);
    StressResult run_soak(const StressConfig& cfg);
    StressResult run_spike(const StressConfig& cfg);
    void run_chaos(const string& server_url);

private:
    // Ogni worker simula un utente
    void worker(const string& url, const string& prompt,
                int max_tokens, atomic<bool>& running,
                StressResult& result, mutex& mtx);
};

} // namespace serius::stress
```

---

## 3. Report Automatico

```text
═══════════════════════════════════════════════════════
  STRESS TEST REPORT — 2026-03-05 02:00:00
═══════════════════════════════════════════════════════

  Model:           SeriusModel-7B-Q4
  GPU:             NVIDIA RTX 4090 (24 GB)
  Test Type:       Ramping Load (1→128 users)
  Duration:        8 min

  ┌──────────┬────────┬──────────┬──────────┬────────┐
  │ Users    │ TPS    │ P50 (ms) │ P99 (ms) │ Errors │
  ├──────────┼────────┼──────────┼──────────┼────────┤
  │ 1        │ 84.2   │ 12       │ 15       │ 0      │
  │ 4        │ 312.5  │ 13       │ 18       │ 0      │
  │ 8        │ 580.1  │ 14       │ 25       │ 0      │
  │ 16       │ 890.3  │ 18       │ 45       │ 0      │
  │ 32       │ 1120.0 │ 29       │ 120      │ 0      │
  │ 64       │ 1180.5 │ 54       │ 850      │ 12     │
  │ 128      │ 1150.2 │ 111      │ 4200     │ 89     │
  └──────────┴────────┴──────────┴──────────┴────────┘

  ✅ Peak throughput: 1180 TPS @ 64 users
  ⚠️ P99 > 1s above 64 users
  ❌ Errors start at 64 concurrent users

  VRAM: stable at 5.2 GB (no leak detected)
  RAM:  stable at 2.1 GB (no leak detected)
═══════════════════════════════════════════════════════
```

---

## 4. CLI

```bash
# Ramping load test
serius stress --type ramp --max-users 64 --duration 300 \
    --model model.gguf --output stress_report.json

# Soak test (4 ore)
serius stress --type soak --users 8 --duration 14400 \
    --model model.gguf --output soak_report.json

# Chaos test
serius stress --type chaos --server http://localhost:8080
```

---

## 5. Struttura File

| File | Contenuto |
| --- | --- |
| `src/stress/stress_runner.h/.cpp` | Runner per ramp/soak/spike/chaos |
| `src/stress/worker.h/.cpp` | Worker thread (simula utente) |
| `src/stress/report.h/.cpp` | Report formattato (ASCII table + JSON) |
| `src/stress/leak_detector.h/.cpp` | Trend analysis VRAM/RAM |
| `src/cli/cmd_stress.cpp` | Comando CLI `serius stress` |

---

## 6. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_ramp_basic` | TPS sale con utenti, plateau a saturazione |
| `test_soak_no_leak` | 1h di carico → VRAM drift < 1 MB |
| `test_spike_recovery` | TPS torna a baseline dopo spike |
| `test_chaos_no_crash` | Input malformati → errori graceful, no crash |
| `test_report_format` | Report JSON e ASCII table validi |
