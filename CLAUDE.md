# Data Mining e Machine Learning — Guida di Studio

Questo repository contiene il materiale del corso **Data Mining e Machine Learning**
(Master DISD, Prof. Pierpaolo Brutti). Questo file serve come mappa di
navigazione dei file; i contenuti teorici veri e propri sono in due documenti
dedicati, da consultare per primi:

- **`SINTESI_CORSO.md`** — sintesi teorica per argomento (workflow di ML,
  regressione/multicollinearità, Ridge/Lasso, PCA/PCR/PLS, classificazione,
  alberi/ensemble, SVM/kernel trick, reti neurali, `caret`), con formule e
  riferimenti a ISLR.
- **`PIANO_STUDI.md`** — percorso di studio in 7 blocchi con esercizi
  concreti sugli script del corso e domande di autovalutazione.
- **`RIASSUNTO_ISLR.md`** — riassunto completo, capitolo per capitolo, del
  libro di testo `materiale/ISLRv2_corrected_June_2023-1.pdf` (tutti e 13 i
  capitoli; quelli coperti dal corso — 2,3,4,5,6,8,9 — più approfonditi, gli
  altri — 7,10-13 — più sintetici).
- **`NOTE_SESSIONE_FEATURE_MAP_RIDGE_LASSO.md`** — note di una sessione di
  Q&A: spiegazioni intuitive (con analogie) di feature map (Φ) e kernel
  trick, riepilogo del Day 1 (`meatspec`), Ridge/Lasso spiegati a partire
  dall'overfitting osservato nel Day 1, riepilogo del Day 2 (selezione per
  correlazione + k-fold CV handmade), PCA spiegata con l'esempio del cavallo
  (`horse-PCA.pdf`), PCR/PLS spiegati a partire dal Day 2, riepilogo del
  Day 3 (CART/alberi su `GlaucomaMVF`), instabilità degli alberi,
  bagging/random forest/boosting spiegati come due logiche opposte (media di
  pareri indipendenti vs correzione sequenziale del bias), e il pruning
  (`cp`) come lo stesso "guinzaglio" di λ e del cutoff del Day 2. Non
  sostituisce `SINTESI_CORSO.md` (più formale/completo) ma è utile come
  spiegazione "in parole semplici" di raccordo tra quei concetti.

Come usarlo: chiedi a Claude cose come "spiegami la PCR partendo dal codice di
day02", "interrogami sugli alberi CART", "confronta bagging e random forest",
ecc. Claude leggerà i file sorgenti indicati sotto (e in `SINTESI_CORSO.md`)
e userà anche conoscenza generale/online per contestualizzare.

## Come sono organizzati i file

Tutto il materiale ufficiale e finale del corso è in
`materiale/Fine corso/`, sotto forma di zip **già estratti** in sottocartelle
(a volte annidate, per via di zip dentro zip nel materiale originale del
docente):

- `Fine corso/Giorno 1 Prof. Brutti-20260602/day01/day01/` → `day01.R`
  (+ `day01.html`, `meatspec.txt`, ecc.)
- `Fine corso/Giorno 2 Prof. Brutti-20260602/day02/day02/` → `day02.R`
  (+ `day02.html`); nella cartella padre anche `horse-PCA.pdf`
- `Fine corso/Giorno 3 Prof. Brutti (online)-20260602/` → `CART-screen.R`,
  `CART-screen2.R`, `CART-screen3.R`, `CART-plot.R`, `glaucoma-data.RData`,
  file `YouTube Links`
- `Fine corso/Giorno 4+5 Prof. Brutti-20260602/day04-05/day04-05/` →
  `day04-05.R`, `DISD - Classifiers Zoo.R`, `DISD - Kernel Trick.R`
  (+ relativi `.html` compilati)
- `Fine corso/Lavagnate-20260602/` → `day2_lavagnate.pdf`,
  `day4_lavagnate.pdf`, `day5_lavagnate.pdf` — **foto della lavagna**:
  contengono la formalizzazione matematica (notazione generale, Ridge/Lasso,
  Naive Bayes, bias-variance in RF/GB) che non è nel codice R. Sono PDF di
  sole immagini: leggibili con il tool Read (rendering diretto), non
  richiedono OCR esterno.

Altri dataset per esercitazione (fuori da "Fine corso"), anch'essi già
estratti:
- `materiale/network/network/` — `netintru.txt`, `network.pdf`,
  `network_info.pdf` (NSL-KDD, network intrusion detection)
- `materiale/buzz/buzz/` — `buzzdata.txt`, `buzz_info.txt`, `buzz.pdf`
  (Buzz in social media, classificazione binaria)

I file `*.html` sono le versioni "compilate" degli script (con output e
grafici già renderizzati). I file `.RData`/`.Rdata` sono workspace salvati di
R, non leggibili come testo: utili solo se l'utente esegue effettivamente R.

## Testo di riferimento

- `materiale/ISLRv2_corrected_June_2023-1.pdf` — **An Introduction to
  Statistical Learning, 2nd ed.** (James, Witten, Hastie, Tibshirani). Libro
  di riferimento per quasi tutti gli argomenti del corso. Sito ufficiale col
  PDF gratuito e i lab in R/Python: https://www.statlearning.com/

## Risorse online consigliate

- **Documentazione `caret`**: https://topepo.github.io/caret/
- **CRAN Task View: Machine Learning**: https://cran.r-project.org/web/views/MachineLearning.html
- **StatQuest (YouTube)** di Josh Starmer — spiegazioni intuitive di alberi,
  random forest, boosting, PCA, SVM, reti neurali.
- I 3 video linkati dal docente su alberi di classificazione (file
  `YouTube Links` nella cartella Giorno 3).

## Note per Claude quando aiuta a studiare

- Per il contenuto teorico, usa sempre prima `SINTESI_CORSO.md` (già
  organizzato per argomento e collegato ai file sorgente) invece di rileggere
  tutti gli script da zero.
- Quando l'utente chiede una spiegazione teorica, aggancia sempre la
  spiegazione al codice/dataset concreto (es. `meatspec` per PCR/PLS/Ridge),
  citando il capitolo ISLR corrispondente.
- Le "Lavagnate" contengono spesso la formalizzazione matematica di argomenti
  presenti nel codice solo in forma implicita (es. Ridge/Lasso non hanno
  script R dedicato, solo teoria alla lavagna) — utile citarle quando la
  domanda dell'utente è più teorica/formale che applicativa.
- Se il repository viene modificato ancora (nuovo materiale, nuovi zip),
  aggiorna prima questa mappa e poi propaga le modifiche a `SINTESI_CORSO.md`
  e `PIANO_STUDI.md`.
