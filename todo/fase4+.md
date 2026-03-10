# Fase 4+ — Cose mancanti (Tokenizer e pre-processing)

Riferimento: `fase4_tokenizer.md`

---

## 1. BPE

| Elemento | Stato | Nota |
|----------|--------|-----|
| Training BPE (merge per frequenza) | Da verificare | bpe.cpp/h |
| encode(text) → token IDs | Da verificare | |
| decode(ids) → testo | Da verificare | |
| Vocabolario + merge rules | ✅ vocab.h citato |
| Special tokens (PAD, BOS, EOS, UNK) | Da verificare | |
| test_bpe_train, test_roundtrip, test_special_tokens | ✅ test_tokenizer.cpp |

---

## 2. DataLoader

| Elemento | Stato | Nota |
|----------|--------|-----|
| Multi-thread (worker pool, queue) | Da verificare | dataloader.cpp/h |
| load_files(paths), next_batch() | Da verificare | |
| shuffle(), pad_batch(), create_mask() | Da verificare | |
| Struct Batch (input_ids, target_ids, padding_mask) | ✅ batch.h |
| Target shift di 1 per LM | Da verificare | |
| test_batch_shape, test_padding, test_target_shift, test_multithread | ✅ test_dataloader.cpp |

---

## 3. Struttura file

| Previsto fase4 | Attuale |
|----------------|---------|
| src/tokenizer/bpe.h, bpe.cpp | ✅ |
| src/tokenizer/vocab.h | ✅ |
| src/data/dataloader.h, .cpp | ✅ |
| src/data/batch.h | ✅ |
| tests/test_tokenizer.cpp, test_dataloader.cpp | ✅ |

---

## 4. Mancante / opzionale

| Elemento | Nota |
|----------|-----|
| FlashTokenizer / tiktoken compat | Fase4 cita tiktoken; progetto usa BPE custom — ok |
| Integrazione con training loop (Batch direttamente da DataLoader) | Da verificare in trainer |

---

**Priorità:** verificare roundtrip encode/decode e correttezza target shift; test multithread senza race.
