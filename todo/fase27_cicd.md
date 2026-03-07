# Fase 27 — CI/CD Pipeline per Modelli Pesanti

> Software tradizionale si testa in secondi; un LLM richiede ore.
> Servono runner GPU-aware e model versioning esterno a Git.

---

## 1. Hardware-Aware Testing

### Pipeline GitHub Actions (con GPU self-hosted)

```yaml
name: LLM CI/CD
on: [push, pull_request]

jobs:
  # Job 1: Build + test CPU (su runner standard)
  build-test-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: cmake -B build -DUSE_CUDA=OFF && cmake --build build -j4
      - name: Unit Tests (CPU)
        run: ctest --test-dir build --output-on-failure
      - name: Math Accuracy Tests
        run: ./build/test_vs_numpy

  # Job 2: Test GPU (su runner self-hosted con GPU)
  test-gpu:
    runs-on: [self-hosted, gpu, linux]
    needs: build-test-cpu
    steps:
      - uses: actions/checkout@v4
      - name: Build with CUDA
        run: cmake -B build -DUSE_CUDA=ON && cmake --build build -j8
      - name: GPU Unit Tests
        run: ctest --test-dir build -R cuda --output-on-failure
      - name: Flash Attention Test
        run: ./build/test_flash_attn
      - name: Benchmark matmul
        run: ./build/bench_matmul --output results.json
      - name: Upload benchmark
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: results.json

  # Job 3: Eval (solo su main, con modello caricato)
  eval:
    runs-on: [self-hosted, gpu, linux]
    if: github.ref == 'refs/heads/main'
    needs: test-gpu
    steps:
      - name: Download model
        run: python scripts/download_model.py --model base-125m --dest /models
      - name: Run MMLU eval
        run: ./build/eval_suite --model /models/base-125m --bench mmlu
      - name: Regression check
        run: python scripts/check_regression.py --baseline evals/baseline.json
```

---

## 2. Model Versioning

### Problema
I pesi del modello sono troppo grandi per Git (> 100 GB). Servono soluzioni esterne.

### Opzioni

| Soluzione | Pro | Contro |
|---|---|---|
| **HuggingFace Hub** | Standard de facto, API Python/CLI | Cloud-only |
| **AWS S3 / GCS** | Economico, veloce | Setup manuale |
| **DVC (Data Version Control)** | Si integra con Git | Complessità extra |
| **Git LFS** | Trasparente | Costoso per file enormi |

### Script di download automatico

```python
# scripts/download_model.py
import argparse, os, hashlib
from huggingface_hub import hf_hub_download

def download_model(model_id, dest):
    files = ["model.safetensors.index.json"]
    # Scarica index
    idx = hf_hub_download(model_id, "model.safetensors.index.json", local_dir=dest)
    # Scarica tutti gli shard
    import json
    with open(idx) as f:
        index = json.load(f)
    shards = set(index["weight_map"].values())
    for shard in shards:
        print(f"Downloading {shard}...")
        hf_hub_download(model_id, shard, local_dir=dest)
    # Verifica checksum
    verify_checksums(dest, index)
```

### Versioning nel codice

```cpp
// model_version.h - generato dalla CI
constexpr const char* MODEL_VERSION = "v1.2.0";
constexpr const char* MODEL_SHA256 = "a1b2c3...";
constexpr const char* TRAINING_DATE = "2026-02-24";
```

---

## 3. Regression Testing

```python
# scripts/check_regression.py
import json, sys

baseline = json.load(open("evals/baseline.json"))
current = json.load(open("evals/current.json"))

regressions = []
for bench in baseline:
    if current[bench] < baseline[bench] - 0.02:  # 2% margin
        regressions.append(f"{bench}: {baseline[bench]:.1%} -> {current[bench]:.1%}")

if regressions:
    print("REGRESSIONS DETECTED:")
    for r in regressions: print(f"  ❌ {r}")
    sys.exit(1)
print("✅ No regressions")
```

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `.github/workflows/ci.yml` | Pipeline CI/CD completa |
| `scripts/download_model.py` | Download automatico da HF Hub |
| `scripts/check_regression.py` | Regression check su eval scores |
| `evals/baseline.json` | Scores di riferimento |

---

## 5. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_ci_cpu` | Build + test CPU passano su runner standard |
| `test_ci_gpu` | Build + test GPU passano su runner GPU |
| `test_download` | Script scarica modello correttamente |
| `test_regression` | Regression check scatta su peggioramento > 2% |
