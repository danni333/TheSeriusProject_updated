# Fase 4 — Tokenizer e Pre-processing (BPE / tiktoken / FlashTokenizer)

> I computer non leggono parole, leggono numeri.
> Il Tokenizer converte testo in sequenze di ID,
> il Data Loader li carica in batch pronti per il modello.

---

## 1. Byte Pair Encoding (BPE)

### Cos'è

BPE parte dai singoli caratteri e iterativamente **fonde** la coppia di simboli più frequente in un nuovo simbolo, fino a raggiungere la dimensione desiderata del vocabolario.

### Algoritmo di Training

```text
vocabolario = {tutti i byte 0-255} + token speciali (<PAD>, <BOS>, <EOS>, <UNK>)
corpus = testo di training splittato in caratteri

Ripeti per num_merges iterazioni:
    1. Conta tutte le coppie adiacenti nel corpus
    2. Trova la coppia (a, b) più frequente
    3. Crea un nuovo token "ab"
    4. Sostituisci ogni occorrenza di (a, b) con "ab" nel corpus
    5. Aggiungi "ab" al vocabolario
    6. Salva la regola di merge: (a, b) → "ab"
```

### Esempio concreto

```text
Testo:  "la casa la cala"
Step 0: ['l','a',' ','c','a','s','a',' ','l','a',' ','c','a','l','a']
Step 1: coppia più frequente ('l','a') → merge → 'la'
        ['la',' ','c','a','s','a',' ','la',' ','c','a','la']
Step 2: coppia più frequente ('c','a') → merge → 'ca'
        ['la',' ','ca','s','a',' ','la',' ','ca','la']
...
```

### Encoding (testo → token IDs)

```python
encode(text):
    tokens = split_to_bytes(text)
    for (a, b) in merge_rules (in ordine di training):
        tokens = replace_all(tokens, (a, b), merged_token)
    return [vocab_to_id[t] for t in tokens]
```

### Decoding (token IDs → testo)

```python
decode(ids):
    return join([id_to_vocab[id] for id in ids])
```

### Parametri configurabili

| Parametro | Tipico | Descrizione |
| --- | --- | --- |
| `vocab_size` | 128,000 | Dimensione vocabolario finale (es. Llama-3 usa 128k, GPT-4 usa 100k con tiktoken) |
| `num_merges` | ~127,744 | `vocab_size - 256 - num_special` |
| `special_tokens` | 4+ | `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`, `&lt;&#124;im_start&#124;&gt;`, `&lt;&#124;im_end&#124;&gt;` |

---

## 2. Data Loader Multi-Thread

### Obiettivo

Leggere file di testo dal disco, tokenizzarli e caricarli in batch pronti per il modello, **senza bloccare il training**.

### Architettura

```text
┌──────────────────────────────────────────────────┐
│                  DataLoader                       │
│                                                   │
│  Thread Pool (N worker threads)                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │ Worker 1 │ │ Worker 2 │ │ Worker 3 │            │
│  │ read()   │ │ read()   │ │ read()   │            │
│  │ tokenize()│ │ tokenize()│ │ tokenize()│           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│       │             │             │                  │
│       ▼             ▼             ▼                  │
│  ┌─────────────────────────────────────┐            │
│  │    Concurrent Queue (batch buffer)  │            │
│  └────────────────┬────────────────────┘            │
│                   │                                  │
│                   ▼                                  │
│         Training Loop (consumatore)                  │
└──────────────────────────────────────────────────────┘
```

### Responsabilità del DataLoader

| Funzione | Descrizione |
| --- | --- |
| `load_files(paths)` | Indicizza tutti i file di testo |
| `next_batch()` | Ritorna il prossimo batch dalla coda (blocca se vuota) |
| `shuffle()` | Mescola l'ordine dei campioni ad ogni epoch |
| `pad_batch()` | Padding a `max_seq_len` con token `<PAD>` |
| `create_mask()` | Maschera che indica quali token sono padding |

### Formato del batch

```cpp
struct Batch {
    Tensor input_ids;    // shape: (batch_size, seq_len)
    Tensor target_ids;   // shape: (batch_size, seq_len) — shifted di 1
    Tensor padding_mask; // shape: (batch_size, seq_len) — 0 dove è pad
};
```

> [!NOTE]
> **Target**: per il language modeling, il target è l'input shiftato di 1 posizione.
> Input: `[<BOS>, il, gatto, è]` → Target: `[il, gatto, è, <EOS>]`

---

## 3. Struttura File

| File | Contenuto |
| --- | --- |
| `src/tokenizer/bpe.h` | Classe `BPETokenizer` |
| `src/tokenizer/bpe.cpp` | Training, encode, decode |
| `src/tokenizer/vocab.h` | Gestione vocabolario e merge rules |
| `src/data/dataloader.h` | Classe `DataLoader` |
| `src/data/dataloader.cpp` | Implementazione multi-thread |
| `src/data/batch.h` | Struct `Batch` |
| `tests/test_tokenizer.cpp` | Unit test BPE |
| `tests/test_dataloader.cpp` | Unit test DataLoader |

---

## 4. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_bpe_train` | Il vocabolario cresce correttamente, le merge sono ordinate per frequenza |
| `test_roundtrip` | `decode(encode(text)) == text` per qualsiasi testo |
| `test_special_tokens` | `<PAD>`, `<BOS>`, `<EOS>` hanno ID riservati |
| `test_batch_shape` | Output shape = `(batch_size, seq_len)` |
| `test_padding` | I token `<PAD>` sono alla fine, la mask è corretta |
| `test_target_shift` | Target è shiftato di 1 rispetto all'input |
| `test_multithread` | DataLoader con 4 thread non produce race condition |
