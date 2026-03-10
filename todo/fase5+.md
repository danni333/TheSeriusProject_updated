# Fase 5+ — Cose mancanti (Training loop)

Riferimento: `fase5_training.md`

---

## 1. Optimizer (AdamW)

| Elemento | Stato | Nota |
|----------|--------|-----|
| m, v (momenti), bias correction, weight decay | Da verificare | adamw.h/cpp |
| Iperparametri lr, β1, β2, ε, λ | Da verificare | |
| ParamState per parametro | Da verificare | |
| Muon optimizer (alternativa SOTA) | ❌ | Opzionale, citato in fase5 |
| test_adamw_step, test_adamw_convergence, test_weight_decay | Da verificare in test_training / test_adamw |

---

## 2. Learning rate scheduler

| Elemento | Stato |
|----------|--------|
| Warmup + stable + decay (WSD) | Da verificare in lr_schedule |
| Warmup lineare + cosine (classico) | Da verificare |
| get_lr(step) senza max_steps obbligatorio (WSD) | Da verificare |
| test_lr_schedule | Da verificare |

---

## 3. Checkpointing

| Elemento | Stato |
|----------|--------|
| Formato .ckpt (magic, version, params, name/shape/data) | Da verificare checkpoint.h/cpp |
| save_checkpoint(model, optimizer, epoch, step) | Da verificare |
| load_checkpoint → ripristino modello + optimizer state | Da verificare |
| test_checkpoint_roundtrip (save → load → save identico) | Da verificare |

---

## 4. Training loop

| Elemento | Stato |
|----------|--------|
| Forward → backward → gradient clipping → optimizer step → zero_grad | Da verificare trainer.cpp |
| clip_grad_norm(params, max_norm) | Da verificare |
| Log (loss, lr, step) | Da verificare |
| Salvataggio checkpoint ogni N epoch | Da verificare |
| Validation set (opzionale) | Da verificare |
| test_gradient_clipping, test_training_loop (loss decresce) | Da verificare |

---

## 5. OpenMP

| Elemento | Stato |
|----------|--------|
| find_package(OpenMP), link OpenMP | ✅ in src/CMakeLists |
| #pragma omp in matmul / elementwise / dataloader | Da verificare dove applicato |

---

## 6. File

| File | Stato |
|------|--------|
| src/optim/adamw.h, .cpp | ✅ |
| src/optim/lr_schedule.h, .cpp | ✅ |
| src/train/trainer.h, .cpp, checkpoint.h, .cpp | ✅ |
| test_adamw, test_checkpoint, test_training | test_training.cpp presente; verificare copertura |

---

**Priorità:** test convergenza AdamW e roundtrip checkpoint; WSD scheduler se non presente; gradient clipping verificato.
