# Fase 15 — Local Model Import: GGUF & Safetensors (Llama-3, Qwen, Mistral)

> Modelli enormi richiedono file multipli e un formato sicuro e veloce.

---

## 1. Importare modelli esistenti (Ecosistema Locale)

L'obiettivo di TheSeriusProject è poter eseguire **immediatamente** modelli di frontiera già addestrati (es. Llama-3-8B, Mistral, Qwen) sul proprio PC, senza script Python. Si devono perciò supportare gli standard `GGUF` e `Safetensors`.

---

## 2. Checkpoint Sharding Server-side (Safetensors)

### Problema

Un modello 70B in FP16 = **140 GB**. Un singolo file è lento e ingestibile.

### Soluzione: shard multipli

```
model/
├── model-00001-of-00008.safetensors   (17.5 GB)
├── ...
├── model-00008-of-00008.safetensors   (17.5 GB)
└── model.safetensors.index.json       (indice: peso → file)
```

### Caricamento parallelo

```cpp
void load_sharded(const string& dir, Model& model) {
    auto index = json::parse(read_file(dir + "/model.safetensors.index.json"));
    // Raggruppa per shard, carica ogni shard in un thread separato
    vector<thread> workers;
    for (auto& [shard, weights] : group_by_shard(index))
        workers.emplace_back([&]{ load_shard(dir + "/" + shard, weights, model); });
    for (auto& t : workers) t.join();
}
```

### Salvataggio shardato

Divide i parametri in shard da `max_shard_size` (es. 5 GB) e scrive l'index JSON.

---

## 2. Formato Safetensors

### Perché non pickle

I `.pkl` Python possono **eseguire codice arbitrario**. Safetensors è puramente dati.

### Struttura file

```
[8 byte: header_size (uint64 LE)]
[Header JSON: nome → {dtype, shape, data_offsets}]
[Data blob: tensori raw contigui]
```

### Parser C++ con mmap (Zero-Copy)

```cpp
class SafetensorsFile {
    void* mapped_;
    size_t file_size_;
    json header_;
    const uint8_t* data_start_;
public:
    SafetensorsFile(const string& path) {
        // Memory-map il file intero
        // Windows: CreateFileMapping + MapViewOfFile
        // Linux: mmap()
        uint64_t header_size = *(uint64_t*)mapped_;
        header_ = json::parse((char*)mapped_ + 8, header_size);
        data_start_ = (uint8_t*)mapped_ + 8 + header_size;
    }
    Tensor get_tensor(const string& name) {
        auto& info = header_[name];
        size_t start = info["data_offsets"][0];
        float* ptr = (float*)(data_start_ + start);
        return Tensor(ptr, info["shape"], /*owns=*/false); // zero-copy!
    }
};
```

### Vantaggi mmap vs fread

| | `fread` | `mmap` |
|---|---|---|
| Copia dati | Sì | No (zero-copy) |
| Startup | Lento | Istantaneo |
| RAM | Alloca tutto | On-demand |

---

## 4. GGUF: Il formato per il Consumer Locale (llama.cpp)

A differenza di Safetensors che di solito contiene pesi float 16-bit splittati in chunk (per GPU-cluster), **GGUF** (GPT-Generated Unified Format) contiene l'intero modello già **quantizzato (es. 4-bit, 5-bit, 8-bit)** in un singolo file monolitico, completo di architettura e configurazione tokenizer.

### Parsing Mmap

Un file GGUF contiene:

1. `Magic Number` ('GGUF')
2. `Version`, `Tensor Count`, `Metadata KV Count`
3. `Key-Value Pairs`: parametri di rete astratti (es. `llama.attention.head_count`) in standard schema.
4. `Tensor Infos`: {name, shape, offset, quant_type}
5. `Data Blobs`: pesi raw già allineati (alignment=32) per mmap.

### Ibrido On-the-Fly

TheSeriusProject dovrà parsare GGUF, leggere l'architettura in run-time configurando dinamicamente il Multi-head Latent Attention / Transformer Block, e passare i puntatori Memory-Mapped direttamente all'Hardware Abstraction Layer (Fase 6) affinché il layer Quantizzazione (Fase 7) li decodifichi ad execution-time durante l'Attention.

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/io/safetensors.h/.cpp` | Parser Safetensors (mmap, zero-copy) |
| `src/io/safetensors_writer.h/.cpp` | Writer Safetensors |
| `src/io/sharding.h/.cpp` | Load/save shardato con index |
| `src/io/mmap.h` | Wrapper cross-platform mmap/MapViewOfFile |
| `tests/test_safetensors.cpp` | Parse, round-trip, dtype |
| `tests/test_sharding.cpp` | Sharding, parallel load |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_safetensors_parse` | Header e dati letti correttamente |
| `test_zero_copy` | Tensor punta a memoria mappata |
| `test_write_read` | Scrivi → leggi: dati identici |
| `test_sharding` | 4 shard → ricarica → pesi identici |
| `test_parallel_load` | 4 thread senza race condition |

### Benchmark

| Benchmark | Misura |
|---|---|
| `bench_load_time` | mmap vs fread per file 1 GB |
| `bench_parallel` | Speedup: 1 vs 4 vs 8 thread |
