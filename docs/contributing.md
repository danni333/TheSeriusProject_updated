# Contributing

Grazie per il tuo interesse a contribuire a **TheSeriusProject**!

---

## 1. Principi

- **Chiarezza prima di tutto**: il codice dovrebbe essere leggibile almeno quanto è veloce.
- **Test‑first mindset**: ogni modifica che cambia il comportamento dovrebbe essere coperta da test.
- **Performance misurata**: le ottimizzazioni dovrebbero essere accompagnate da benchmark prima/dopo.

---

## 2. Workflow di base

1. Fai un **fork** del repository.
2. Crea un branch:
   - `feature/<nome>` per nuove funzionalità.
   - `fix/<nome>` per bugfix.
3. Implementa la modifica, mantenendo lo stile esistente.
4. Esegui i test:

   ```bash
   mkdir -p build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
   ctest --output-on-failure
   ```

5. Apri una **pull request** descrivendo:
   - il problema che risolvi o la feature che introduci;
   - l'approccio scelto;
   - i risultati di test/benchmark rilevanti.

---

## 3. Stile del codice

- C++17 o superiore.
- Evitare dipendenze esterne non strettamente necessarie.
- Tenere separata la logica core dal codice specifico di piattaforma quando possibile.

---

## 4. Documentazione

- Aggiorna la documentazione in `docs/` ogni volta che:
  - aggiungi una nuova funzionalità utente,
  - cambi l'interfaccia pubblica (CLI, API),
  - modifichi in modo rilevante il comportamento di componenti chiave.

Se aggiungi un nuovo modulo importante, considera di aggiornare anche `docs/architecture.md`.

---

## 5. Issue e feature request

Quando apri una issue, includi:

- versione di sistema operativo, compilatore, CMake, CUDA (se rilevante);
- passi minimi per riprodurre il problema;
- log e output di test/benchmark pertinenti.

Per proposte di nuove funzionalità, collega se possibile la proposta alla roadmap (file in `todo/`).

