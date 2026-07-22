# Piano di studi — Data Mining e Machine Learning

Piano organizzato in **7 blocchi tematici**, pensato per essere seguito in
sequenza (ogni blocco richiede i concetti del precedente). Ogni blocco indica:
tempo indicativo, cosa leggere/rivedere, cosa eseguire concretamente in R, e
domande di autovalutazione. Riferimenti dettagliati alla teoria in
`SINTESI_CORSO.md`, mappa dei file in `CLAUDE.md`.

Ritmo suggerito: **~2-3 settimane**, 3-4 sessioni di studio a settimana (~1h30
ciascuna). I blocchi 1-5 sono il nucleo imprescindibile del corso; i blocchi
6-7 possono essere compressi se hai meno tempo.

**Nota sui percorsi**: dopo l'estrazione degli zip in `materiale/Fine corso/`,
i file vivono in sottocartelle annidate (es.
`materiale/Fine corso/Giorno 1 Prof. Brutti-20260602/day01/day01/day01.R`).
I percorsi qui sotto sono quelli reali post-estrazione.

Setup una tantum prima di iniziare:
```r
install.packages(c("caret", "corrplot", "viridis", "pls",
                    "C50", "gbm", "ipred", "party", "partykit",
                    "rattle", "rpart", "rpart.plot", "RColorBrewer", "RWeka",
                    "mlbench", "e1071", "MASS", "nnet", "randomForest", "plotly",
                    "glmnet"))
```
(Nota: `RWeka` richiede Java installato sul sistema. `glmnet` non è usato
esplicitamente negli script del corso ma è la libreria R standard per
esercitarsi su Ridge/Lasso — vedi Blocco 2.)

---

## Blocco 1 — Fondamenta: framework statistico e regressione lineare
**Tempo: ~2 sessioni**

**Studia**
- `SINTESI_CORSO.md` §1 (impostazione formale: 𝒟ₙ, f_θ(x)=θᵀΦ(x), ERM,
  K-fold CV) e §2 (regressione lineare e multicollinearità)
- `materiale/Fine corso/Lavagnate-20260602/day2_lavagnate.pdf` (le due foto
  della lavagna: notazione generale + schema K-fold CV)
- ISLR cap. 2 (bias-variance trade-off) e cap. 3 (regressione lineare)

**Esegui**
- Apri/esegui riga per riga `materiale/Fine corso/Giorno 1 Prof. Brutti-20260602/day01/day01/day01.R`
  (dataset `meatspec.txt`), osservando ogni grafico.
- Ripeti tu stesso, senza guardare lo script: import dati, istogramma della
  risposta, split train(70%)/test(30%) con `set.seed()`, `lm(fat~., data=tr)`,
  RMSE in train e test, commento su overfitting.
- Continua con `materiale/Fine corso/Giorno 2 Prof. Brutti-20260602/day02/day02/day02.R`:
  prova `findCorrelation()` con almeno 3 cutoff diversi e ricostruisci a mano
  la curva "errore train vs K-fold CV" in funzione del numero di variabili.

**Autovalutazione**
- Cosa rappresenta la feature map Φ(x) nella formula generale f_θ(x)=θᵀΦ(x)?
  Sapresti dire qual è Φ nel caso della regressione lineare semplice?
- Perché coefficienti "strani" in `summary(lm(...))` non implicano che il
  modello preveda male?
- Perché l'errore di training è sempre una stima ottimistica dell'errore di
  generalizzazione?

---

## Blocco 2 — Metodi penalizzati: Ridge e Lasso
**Tempo: ~1 sessione (teoria + esercizio pratico da costruire tu)**

Questo blocco è stato trattato dal docente **solo alla lavagna** (non c'è
codice R nel materiale del corso): è un buon esercizio implementarlo tu
stesso su `meatspec`, il dataset ideale visto il problema di
multicollinearità già discusso nel Blocco 1.

**Studia**
- `SINTESI_CORSO.md` §3 (Ridge/Lasso: norme Lp, formulazione vincolata vs
  penalizzata, soluzione chiusa della Ridge, sparsità del Lasso)
- `materiale/Fine corso/Lavagnate-20260602/day4_lavagnate.pdf` (le 4 foto
  della lavagna: norme, "palle" Lp, forma chiusa Ridge, Lasso path)
- ISLR cap. 6.2 (Ridge Regression and the Lasso)

**Esegui**
- Su `meatspec` (train/test già creati nel Blocco 1), usa `caret::train(...,
  method="glmnet")` oppure il pacchetto `glmnet` direttamente:
  1. adatta una Ridge (`alpha=0`) e un Lasso (`alpha=1`) con CV per scegliere
     λ (`cv.glmnet`)
  2. confronta l'RMSE in test con quello della regressione lineare completa
     e con quello ottenuto da `findCorrelation` (Blocco 1)
  3. guarda quanti coefficienti il Lasso azzera esattamente: è coerente con
     quanto visto nella lavagnata sulla sparsità?
  4. plotta il "regularization path" (`plot(glmnet_fit)`) e confrontalo
     mentalmente col disegno del docente

**Autovalutazione**
- Perché la penalità L1 produce soluzioni sparse e la L2 no? (ragiona sulla
  forma geometrica delle "palle" Lp)
- Nella formula θ̂(λ) = argmin ‖y-𝒳θ‖² + λ‖θ‖, cosa succede al bias e alla
  varianza della stima quando λ→0? E quando λ→∞?
- Perché la Ridge ha soluzione in forma chiusa e il Lasso no?

---

## Blocco 3 — Riduzione dimensionale: PCA, PCR, PLS
**Tempo: ~1-2 sessioni**

**Studia**
- `SINTESI_CORSO.md` §4
- `materiale/Fine corso/Giorno 2 Prof. Brutti-20260602/horse-PCA.pdf`
  (intuizione geometrica: direzioni principali d₁, d₂, d₃ come proiezioni
  successive che massimizzano la varianza catturata)
- ISLR cap. 6.3 (Dimension Reduction Methods)

**Esegui**
- Continua/ripeti la seconda parte di `day02/day02/day02.R`:
  1. `princomp()`/`prcomp()` su `tr[,1:100]`, `summary()` e `plot()` per la
     varianza spiegata
  2. `caret::train(..., method="pcr", tuneLength=30, trControl=trainControl(
     method="cv", number=3))`, guarda `mod_pcr$bestTune` e il plot dell'RMSE
     al variare del numero di componenti
  3. ripeti con `method="pls"`
  4. confronta PCR, PLS, il modello lineare "ridotto" (Blocco 1) e Ridge/
     Lasso (Blocco 2) con `resamples()` + `bwplot(metric="RMSE")` — un
     confronto a 4/5 modelli, più ricco di quello originale del corso

**Autovalutazione**
- Guardando il disegno del cavallo: perché d₂ deve essere ortogonale a d₁?
- Qual è la differenza sostanziale tra come PCA/PCR e PLS scelgono le
  componenti?
- In questo dataset, PLS batte PCR a parità di numero di componenti: perché
  ha senso che sia così?

---

## Blocco 4 — Classificazione: formalizzazione e classificatori di base
**Tempo: ~2 sessioni**

**Studia**
- `SINTESI_CORSO.md` §5 (formalizzazione: perdita 0/1, surrogate losses,
  oltre l'accuracy) e §6 (Naive Bayes formalizzato + tabella zoo classificatori)
- `materiale/Fine corso/Lavagnate-20260602/day5_lavagnate.pdf`, prime 2
  "coppie" di foto (formalizzazione classificazione + Naive Bayes/alberi)
- ISLR cap. 4 (Classification: logistic, LDA, QDA, confronto KNN)

**Esegui**
- `materiale/Fine corso/Giorno 4+5 Prof. Brutti-20260602/day04-05/day04-05/day04-05.R`,
  prima parte: discretizza `fat` sia con `cut()` che con `kmeans(centers=2)`,
  confronta le due discretizzazioni con `table()`. Split stratificato con
  `createDataPartition()`.
- `.../day04-05/DISD - Classifiers Zoo.R`: esegui l'intero script sui dati
  sintetici a cerchi concentrici. Per ciascun classificatore (KNN a k
  diversi, Naive Bayes, LDA, QDA, logistica) osserva il grafico del confine
  di decisione e **descrivi a parole tue** perché ha quella forma.

**Autovalutazione**
- Perché la perdita 0/1 non è quasi mai usata direttamente per *ottimizzare*
  un modello, anche se è quella che vogliamo minimizzare concettualmente?
- Perché LDA e regressione logistica danno confini simili (entrambi lineari)
  ma QDA no?
- Con k=1 nel KNN il confine è molto irregolare: è overfitting o
  underfitting? E con k=10?
- Nella formula del Naive Bayes, cosa renderebbe "meno naive" il modello (e
  perché in pratica si accetta comunque questa semplificazione)?

---

## Blocco 5 — Alberi ed Ensemble (CART, C4.5/C5.0, Bagging, Random Forest, Boosting)
**Tempo: ~3 sessioni (il blocco più corposo)**

**Studia**
- `SINTESI_CORSO.md` §7, §8, §9
- `materiale/Fine corso/Lavagnate-20260602/day5_lavagnate.pdf`, ultime 2
  "coppie" di foto (alberi come partizioni ricorsive; "Back in the days" —
  RF vs GB in ottica bias-variance, Breiman/bootstrap/OOB)
- ISLR cap. 8 (Tree-Based Methods)
- I 3 video linkati dal docente (file `YouTube Links` nella cartella Giorno 3)
  su alberi di classificazione

**Esegui**, in ordine, sul dataset `GlaucomaMVF`
(`materiale/Fine corso/Giorno 3 Prof. Brutti (online)-20260602/`):
1. `CART-screen.R`: `rpart()` + lettura `cptable`, scelta del `cp` ottimo,
   `prune()`; `J48()` (C4.5); `C5.0()` semplice + `C5imp()`; `ctree()`
   (party); confronta tutti su train/test error.
2. `CART-plot.R` (dataset `caret::segmentationData`): confronta un "big tree"
   (`rpart.control(minsplit=20, cp=0)`, deliberatamente in overfitting) con
   un albero di default, usando `prp()` (rpart.plot) e `fancyRpartPlot()`
   (rattle). Prova anche il pruning interattivo `prp(tree, snip=TRUE)`.
3. `CART-screen2.R`: `ipred::bagging()` con stima OOB; **implementazione a
   mano del bagging** (bootstrap con `rmultinom`, 25 alberi, media delle
   predizioni OOB) — capiscila riga per riga; `randomForest()` +
   `importance()`/`varImpPlot()`; `gbm()` e `C5.0(trials=10)` per il boosting.
4. `CART-screen3.R`: `RWeka::PART()` e `C5.0(rules=TRUE)` (regole);
   `caret::train(method="rpart")` come wrapper unificato.
5. Torna a `day04-05.R` (seconda parte, sezione "Classifichiamo"): questa
   volta lo script **risolve per intero** l'esercizio su `meatclass` con un
   albero (`train(method="rpart2")`, tuning su `maxdepth`, `varImp`,
   `fancyRpartPlot`, `confusionMatrix`, `resamples`, `twoClassSummary`/ROC,
   `repeatedcv`). Studialo a fondo, poi **completa tu l'esercizio lasciato
   aperto dal docente**: adatta una Random Forest sugli stessi dati e
   confrontala col miglior albero (stesso procedimento: `varImp`,
   `confusionMatrix`, `resamples`).

**Autovalutazione**
- Perché il docente dice "se il vostro albero è diverso dal mio siatene
  felici"? Cosa dimostra questa instabilità?
- Nel framing "Back in the days": perché RF parte da alberi in overfitting e
  GB da alberi in underfitting? Cosa controlla l'ensembling nei due casi
  (varianza vs bias)?
- A cosa serve il parametro `mtry` nella Random Forest, oltre al bootstrap
  del bagging?
- Che vantaggio pratico offre un classificatore a regole (PART, C5.0 rules)
  rispetto a un albero o una random forest?
- Cos'è il Cohen's Kappa e perché può essere più informativo della sola
  accuracy?

---

## Blocco 6 — SVM, Kernel Trick, cenni di reti neurali
**Tempo: ~1-2 sessioni**

**Studia**
- `SINTESI_CORSO.md` §11, §12
- ISLR cap. 9 (Support Vector Machines)

**Esegui**
- `.../day04-05/DISD - Kernel Trick.R`: riproduci il grafico 2D dei punti non
  separabili, poi il grafico 3D interattivo (`plotly`) dopo aver aggiunto la
  feature `x1^2 + x2^2`. Verifica visivamente che nel 3D i dati siano ora
  separabili da un piano. Ricollega questa mappatura esplicita al concetto
  generale di feature map Φ(x) visto nel Blocco 1.
- Torna allo Zoo dei classificatori e confronta i 4 kernel SVM (`linear`,
  `radial`, `polynomial`, `sigmoid`) sugli stessi dati a cerchi: quale ha
  senso usare e perché?
- Rivedi la parte NN dello Zoo (`nnet(size=1/2/4/10)`), osservando come
  cambia il confine di decisione al crescere delle unità nascoste.

**Autovalutazione**
- Nell'esempio del kernel trick, perché aggiungere `x1²+x2²` come feature
  rende il problema linearmente separabile?
- Qual è il "trick" vero e proprio nelle SVM (cosa si evita di calcolare
  esplicitamente)?
- Perché una rete con troppe unità nascoste rischia overfitting sugli stessi
  identici dati 2D usati per gli altri modelli?

---

## Blocco 7 — Applicazione integrata su dataset nuovi
**Tempo: ~2 sessioni**

Obiettivo: applicare l'intero workflow (Blocchi 1-5) senza traccia, su
dataset del corso non ancora usati in questo piano.

**Esercizio A — Network Intrusion Detection (NSL-KDD)**
Dataset: `materiale/network/network/netintru.txt` (leggi anche
`network_info.pdf` per la descrizione delle 41 feature + `Class` a 5 livelli
+ `Difficulty Level`).
1. Esplora il dataset (classi `normal/dos/probe/r2l/u2r`, verifica se
   sbilanciate con `table()`/`prop.table()`)
2. Split stratificato train/test
3. Fit di almeno 3 modelli visti nei blocchi precedenti (es. albero singolo,
   random forest, boosting) con `caret::train` + CV
4. Confronto finale con `confusionMatrix()` per ciascuna classe (occhio alle
   classi rare tipo `u2r`: l'accuratezza globale può essere fuorviante — usa
   anche Kappa e sensitivity/specificity per classe) e `resamples()`/
   `bwplot()` per il confronto tra modelli

**Esercizio B — Buzz in Social Media**
Dataset: `materiale/buzz/buzz/buzzdata.txt` (leggi `buzz_info.txt` per la
descrizione delle feature e `buzz.pdf`). È un problema di **classificazione
binaria** (`Buzz` = 1/0): 96 feature numeriche = 12 grandezze (numero
discussioni create, "burstiness", numero autori, ecc.) misurate su 8 istanti
temporali consecutivi → aspettati **forte correlazione tra feature** dello
stesso tipo a istanti vicini, la stessa sfida di multicollinearità vista su
`meatspec` ma in un problema di classificazione.
1. Esplora la struttura di correlazione tra feature (stesso approccio del
   Blocco 1: `cor()`, `corrplot`, `findCorrelation`)
2. Split stratificato, vista la probabile natura sbilanciata di `Buzz`
3. Applica selezione di variabili o riduzione dimensionale (PCA) prima del
   classificatore, oppure Ridge/Lasso in versione logistica
   (`glmnet(family="binomial")`) per gestire la ridondanza tra feature
4. Confronta almeno un modello lineare-regolarizzato e un ensemble (RF o
   boosting), con attenzione a sensitivity/specificity oltre all'accuracy

**Autovalutazione finale (ripasso generale)**
- Sapresti spiegare l'intero workflow da dataset grezzo a modello finale
  scelto, citando dove entra in gioco la cross-validation?
- Per ciascuna famiglia di modelli vista (lineari penalizzati, PCR/PLS,
  alberi singoli, ensemble, SVM, NN), sapresti dire in una frase qual è il
  suo principale compromesso bias-variance?
- Su un dataset con classi fortemente sbilanciate, quali accortezze in più
  servono rispetto al workflow "standard"?

---

## Checklist riassuntiva

- [ ] Blocco 1 — Framework statistico + regressione lineare + multicollinearità
- [ ] Blocco 2 — Ridge e Lasso (norme, forma chiusa, sparsità)
- [ ] Blocco 3 — PCA / PCR / PLS
- [ ] Blocco 4 — Classificazione: formalizzazione + Naive Bayes/LDA/QDA/logistica/KNN
- [ ] Blocco 5 — Alberi (CART, C4.5, C5.0) ed ensemble (Bagging, RF, Boosting)
- [ ] Blocco 6 — SVM, kernel trick, cenni NN
- [ ] Blocco 7 — Esercizio integrato su `network` (NSL-KDD) e `buzz`

## Come usare Claude durante lo studio

- Chiedi spiegazioni puntuali agganciate al codice: *"spiegami riga per riga
  la parte di bagging fatto a mano in CART-screen2.R"*.
- Chiedi di formalizzare il codice: *"scrivimi in notazione matematica cosa
  fa `train(method='pcr')` in day02.R, collegandolo alla formula f_θ(x)=θᵀΦ(x)"*.
- Fatti interrogare: *"fammi 5 domande sul Blocco 5 prima di passare oltre"*.
- Chiedi confronti: *"confronta Ridge, PCR e PLS usando i numeri ottenuti nel
  mio script"* (incolla l'output).
- Per l'esercizio RF lasciato aperto in `day04-05.R`, chiedi feedback sul tuo
  codice prima di guardare eventuali soluzioni: *"ho scritto questo codice
  per la RF su meatclass, mancano errori concettuali?"*.
- Se qualcosa nel codice non gira (pacchetto mancante, versione R diversa),
  chiedi aiuto per il debug prima di saltare la sezione.
