# Sintesi del corso — Data Mining e Machine Learning

Master DISD, Prof. Pierpaolo Brutti — 2026.

Sintesi **per argomento** (non per giornata) di tutta la teoria e la pratica
incontrate nel materiale finale del corso (`materiale/Fine corso/`), integrata
con il libro di testo `materiale/ISLRv2_corrected_June_2023-1.pdf` (An
Introduction to Statistical Learning, 2nd ed. — ISLR) e con le note prese alla
lavagna dal docente (`materiale/Fine corso/Lavagnate-20260602/`).

Per il piano di studio scandito nel tempo vedi `PIANO_STUDI.md`. Per la mappa
completa dei file vedi `CLAUDE.md`.

**Nota sulla struttura del materiale**: dopo l'estrazione, ogni giornata del
corso vive sotto `materiale/Fine corso/Giorno N .../` (con eventuali
sottocartelle annidate createsi dall'estrazione degli zip originali). Le tre
"Lavagnate" (foto della lavagna, giorni 2/4/5) sono in
`materiale/Fine corso/Lavagnate-20260602/` e contengono la formalizzazione
matematica che il docente ha scritto a mano durante le lezioni — spesso più
sintetica e rigorosa del codice R, e qui riportata fedelmente.

---

## 1. Impostazione formale di un problema di apprendimento supervisionato

Dalla lavagnata del giorno 2 (`day2_lavagnate.pdf`), la notazione usata per
tutto il corso:

- **y** = variabile risposta; **{x₁,...,x_p}** = variabili esplicative
  (covariate).
- **Dataset**: 𝒟ₙ = {(y₁,x₁), ..., (yₙ,xₙ)} — n unità, p variabili esplicative
  (nell'esempio `meatspec`: p = 100, n = 215).
- **y numerica → Regressione**; **y categoriale → Classificazione**
  (apprendimento **supervisionato** in entrambi i casi: abbiamo etichette/
  risposte osservate su cui allenare il modello).
- **Modello generale a basi/feature map**:

  ```
  f_θ(x₁,...,x_p) = θ₀ + θ₁·φ₁(x) + θ₂·φ₂(x) + ... + θ_K·φ_K(x) = θᵀ·Φ(x)
  ```

  dove **Φ(x) = [φ₁(x),...,φ_K(x)]ᵀ** è una **feature map** (trasformazione
  delle covariate originali), tipicamente con **K ≫ p** (molte più feature
  derivate che covariate originali).
- **Stima dei parametri** (minimi quadrati sul training):

  ```
  θ̂ = argmin_θ  Σ_{i=1}^n ( y_i − f_θ(x_i) )²      [solo su dati TR]
  ```

Questa formulazione è la "chiave di volta" concettuale del corso: **quasi
tutti** i modelli visti sono casi particolari di questo schema, cambiando solo
la scelta della feature map Φ:

| Modello | Feature map Φ(x) |
|---|---|
| Regressione lineare semplice | Φ(x) = x (identità) |
| PCR | Φ(x) = prime K componenti principali di x |
| PLS | Φ(x) = prime K componenti PLS (supervisionate) |
| Kernel trick / SVM non lineare | Φ(x) = mappatura (anche implicita, infinito-dimensionale) in uno spazio dove i dati sono linearmente separabili |
| Reti neurali (shallow) | Φ(x) = output dei neuroni del hidden layer |

Tenere a mente questa "meta-idea" aiuta a vedere il filo conduttore tra
argomenti altrimenti molto diversi tra loro (regressione, riduzione
dimensionale, SVM, reti).

### Cross-validation — schema K-fold

Sempre dalla lavagnata:
- 𝒟ₙ diviso in **training (70%)** e **test (30%)**.
- Sul training, per **ogni valore di una griglia di iperparametri di tuning**,
  si ripete uno schema **K-fold** (esempio K=3):
  - fold 1: Train-Train-Validation
  - fold 2: Train-Validation-Train
  - fold 3: Validation-Train-Train
  - si calcola l'errore su ciascuna Validation (e₁, e₂, e₃) e si riportano
    **media** (`avg(e₁,e₂,e₃)`) e **deviazione standard** (`sd(e₁,e₂,e₃)`) come
    stima (e incertezza sulla stima) dell'errore di generalizzazione.
- Il **test set** resta sempre fuori da questo processo, per una valutazione
  finale onesta del modello scelto.

Riferimento: ISLR cap. 2 (framework statistico dell'apprendimento) e cap. 5
(Cross-Validation, K-fold CV).

---

## 2. Regressione lineare e multicollinearità

Dataset guida: `meatspec` (`materiale/Fine corso/Giorno 1 Prof. Brutti-20260602/day01/day01/day01.R`,
dati in `meatspec.txt`) — 100 predittori spettrali (NIR, funzionalmente molto
regolari) per prevedere il contenuto di grasso (`fat`) di campioni di carne.

- Import, esplorazione, split train(70%)/test(30%) a mano con `sample()` +
  `set.seed()`.
- **Modello**: `lm(fat ~ ., data = tr)`.
- **Sintomo di multicollinearità** (correlazione media assoluta tra le 100
  covariate ≈ 0.98): `summary(mod_all)` mostra coefficienti enormi e di segno
  incoerente — il modello può ancora prevedere ragionevolmente, ma i
  coefficienti **non sono interpretabili**.
- **Segnale di overfitting**: RMSE in test di un ordine di grandezza superiore
  a quello in train usando tutte le 100 variabili.
- **Prima idea di rimedio** (`materiale/Fine corso/Giorno 2 Prof. Brutti-20260602/day02/day02/day02.R`):
  selezione di variabili con `caret::findCorrelation(cor_matrix, cutoff=...)`,
  che rimuove iterativamente le variabili più correlate tra loro — il
  `cutoff` è un vero iperparametro, tarato con **K-fold CV fatta a mano**
  (fold costruiti con `sample(rep(1:K,...))`, doppio ciclo cutoff × fold).
  Nota di attenzione lasciata nel codice: con pochi dati per fold e tante
  variabili (p vicino a n), la stima CV può essere numericamente instabile.
- **Bias-variance trade-off** mostrato empiricamente: l'errore di training
  scende monotonamente al crescere del numero di variabili (troppo
  ottimistico), mentre l'errore di CV ha una forma a U.
- **Nota "avanzata"**: quando p > n (modelli sovraparametrizzati, tipico delle
  reti profonde), si ottimizza con discesa del gradiente; si osserva talvolta
  il fenomeno del **"double descent"** (l'errore in test, dopo il picco nel
  regime d'interpolazione p≈n, torna a scendere).

Riferimento: ISLR cap. 3 (Linear Regression, collinearità).

---

## 3. Metodi penalizzati: Ridge e Lasso

**Argomento trattato interamente alla lavagna** (`day4_lavagnate.pdf`), come
soluzione più elegante al problema di multicollinearità visto sopra (in
alternativa al semplice "buttare via variabili correlate").

**Motivazione**: adattando il modello lineare con tutte le variabili, data
l'alta correlazione tra di esse, le stime dei coefficienti "esplodono" (fuori
controllo — nell'esempio numerico alla lavagna, coefficienti dell'ordine di
±10 mila). Serve una misura di "grandezza" dei coefficienti in θ da tenere
sotto controllo → una **norma di θ**.

**Norme Lp** su θ = (θ₁,...,θ_K):

```
‖θ‖_p = ( Σ_j |θ_j|^p )^(1/p)          in generale
‖θ‖_2 = √(Σ_j θ_j²)                    norma euclidea (L2)
‖θ‖_1 = Σ_j |θ_j|                      norma L1
‖θ‖_∞ = max_j |θ_j|                    L∞
‖θ‖_0 = #{θ_j ≠ 0}                     L0 (conteggio dei coefficienti non nulli)
```

Le "palle" {θ : ‖θ‖_p ≤ 1} hanno forma diversa a seconda di p: per p≥1 sono
tutte **convesse** (cerchio per L2, "diamante"/rombo per L1, quadrato per L∞);
per p<1 (es. L0) **non sono convesse**.

**Formulazione vincolata** ("palla per vincolare la ricerca"):

```
θ̂ = argmin_θ  Σᵢ (yᵢ − f_θ(xᵢ))²      soggetto a  ‖θ‖ ≤ L
```

con **L>0 parametro di tuning**, scelto per **cross-validazione**.

**Forma equivalente penalizzata** (via moltiplicatori di Lagrange):

```
θ̂(λ) = argmin_θ  Σᵢ (yᵢ − f_θ(xᵢ))²  +  λ·‖θ‖
```

con **λ** iperparametro di tuning: **λ↑ ⇒ penalità più forte** (coefficienti
più "ristretti"); L e λ sono in corrispondenza inversa.

- **Ridge Regression (penalità L2)**: `‖θ‖₂²` come penalità. In forma
  matriciale (𝒳 = matrice disegno, y = vettore risposta):

  ```
  θ̂^R(λ) = argmin_θ ‖y − 𝒳θ‖₂² + λ‖θ‖₂²
  ```

  Ha **soluzione in forma chiusa** (derivando e ponendo il gradiente a zero):

  ```
  θ̂^R(λ) = (𝒳ᵀ𝒳 + λ·I)^(-1) 𝒳ᵀy      (λ > 0, "Ridge path" al variare di λ)
  ```

  All'aumentare di λ, i coefficienti si **restringono verso zero** ma
  (tipicamente) **non diventano mai esattamente zero** — soluzione "densa".

- **Lasso Regression (penalità L1)**:

  ```
  θ̂^L(λ) = argmin_θ ‖y − 𝒳θ‖₂² + λ‖θ‖₁
  ```

  La penalità L1 **non è derivabile** in θ=0, quindi niente forma chiusa
  generale ("Lasso path" calcolato con algoritmi dedicati, es. LARS/
  coordinate descent). Il vantaggio pratico: al crescere di λ, **alcuni
  coefficienti diventano esattamente zero** — soluzione **sparsa**, il che
  rende il Lasso anche un metodo di **selezione automatica delle variabili**
  (a differenza della Ridge). Geometricamente: il vertice del "diamante" L1
  cade più facilmente su un asse (coefficiente = 0) rispetto al cerchio L2.

- **Confronto sintetico L2 vs L1** (esempio numerico alla lavagna con K
  coefficienti tutti uguali per costruzione): a parità di norma L2 = 1, un
  vettore "denso" (tutte le componenti uguali, ≈1/√K ciascuna) ha norma L1
  grande (√K); un vettore "sparso" (una sola componente = 1, le altre 0) ha
  norma L1 = 1. Per ottenere sparsità (poche variabili rilevanti) **conviene
  L1**.

- **Riepilogo tassonomia**: L2 → Ridge (anche detta *weight decay* in ambito
  reti neurali); L1 → Lasso; **L0 → subset selection** (selezione esatta del
  sottoinsieme di variabili, ma problema combinatorio **non convesso**, quindi
  computazionalmente intrattabile per p grande) — L1 è vista storicamente come
  il miglior "surrogato convesso" della L0.

Riferimento: ISLR cap. 6.2 (Ridge Regression and the Lasso) — nel libro
trovate anche la derivazione con i moltiplicatori di Lagrange e il confronto
grafico coi contorni delle regioni di vincolo, identico a quanto disegnato
alla lavagna.

---

## 4. Riduzione della dimensionalità: PCA, PCR, PLS

Alternativa alla penalizzazione: invece di scartare/restringere predittori,
**trasformarli** in un numero minore di combinazioni lineari.

- **Intuizione geometrica della PCA** (`materiale/Fine corso/Giorno 2 Prof. Brutti-20260602/horse-PCA.pdf`,
  esempio illustrativo con un modello 3D di cavallo): dato un punto x e
  un'origine x₀, la **prima direzione principale d₁** è quella su cui la
  proiezione `d₁ᵀx` cattura la massima variabilità dei dati; la **seconda
  direzione d₂** è ortogonale a d₁ e cattura la massima variabilità residua
  (proiezione `d₂ᵀy`); e così via per d₃, ..., ottenendo una base ortogonale
  di direzioni via via meno informative in termini di varianza spiegata.
  Questa è esattamente l'idea dietro `princomp`/`prcomp`/`svd` in R.
- **PCA come pre-processing** (`day02.R`): `princomp()`/`prcomp()` sulle sole
  X, in modo *non supervisionato* (non guarda y).
- **PCR (Principal Component Regression)**: regressione lineare sulle prime
  componenti principali. **Attenzione** (nota esplicita nel codice): la
  varianza spiegata da una componente **non garantisce** che sia predittiva
  per y.
- **PLS (Partial Least Squares)**: come la PCR, ma le componenti sono scelte
  massimizzando la **covarianza con la risposta** (supervisionato). Nel corso
  emerge che PLS raggiunge performance comparabili o migliori della PCR con
  **meno componenti**.
- Implementazione via `caret::train(..., method="pcr"/"pls", tuneLength=30,
  trControl=trainControl(method="cv", number=3))`: numero ottimo di
  componenti scelto via CV (`mod$bestTune`).
- **Confronto finale** tra PCR, PLS e regressione lineare "ridotta" con
  `resamples()` + `bwplot(metric="RMSE")`.

Riferimento: ISLR cap. 6.3 (Dimension Reduction Methods: PCR, PLS) — utile
anche il cap. 12 (Unsupervised Learning) per un trattamento più esteso della
PCA in sé.

---

## 5. Classificazione: formalizzazione, discretizzazione, oltre l'accuratezza

### Impostazione formale (lavagnata giorno 5, `day5_lavagnate.pdf`)

- **Classificazione = prevedere una variabile categoriale**; per comodità y
  viene spesso codificata numericamente (es. binaria: `{low, high}` ↔ `{0,1}`
  ↔ `{-1,+1}`). Sia **h_θ(x)** un classificatore.
- **Che metrica/perdita (loss) usare per confrontare y e h_θ(x)?**
  - `(y − h_θ(x))²` è la perdita naturale per la regressione, ma per la
    classificazione si usa tipicamente la **perdita 0/1**:

    ```
    L(θ) = (1/n) Σᵢ 1{ yᵢ ≠ h_θ(xᵢ) }        (1 se diverso, 0 altrimenti)
    θ̂ = argmin_θ L(θ)
    ```
    cioè la proporzione di previsioni sbagliate (**errore di
    classificazione**, esattamente quanto già calcolato con `mean(pred !=
    actual)` negli script).

- **Due osservazioni importanti fatte dal docente:**
  1. In pratica la perdita 0/1 viene sostituita con **"surrogate" più
     semplici da ottimizzare** (non è derivabile) — es. la **hinge loss**
     usata dalle SVM.
  2. La perdita 0/1 valuta solo il **numero totale** di errori commessi →
     **non distingue la "natura" dell'errore** (falso positivo vs falso
     negativo hanno lo stesso peso, anche se in molte applicazioni non sono
     equivalenti — es. diagnosi medica). **Meglio considerare funzioni di
     valutazione più ricche**: sensitivity/specificity, curve ROC/AUC, F1 —
     esattamente ciò che poi si vede in pratica con `caret::confusionMatrix`
     e `twoClassSummary` (vedi §10).

### Da regressione a classificazione (pratica)

`materiale/Fine corso/Giorno 4+5 Prof. Brutti-20260602/day04-05/day04-05/day04-05.R`:
la risposta continua `fat` (meatspec) viene discretizzata in due classi:
- **A occhio**: soglia arbitraria con `cut()` (es. soglia 22).
- **In modo più sistematico**: `kmeans(jnk, centers=2)`, poi si etichettano i
  due cluster come "low"/"high" guardando i centri.
- Split **stratificato** con `caret::createDataPartition()` (classi
  "moderatamente sbilanciate").

Riferimento: ISLR cap. 4 (Classification) — in particolare la sezione sulle
metriche di errore oltre alla semplice accuracy.

---

## 6. Classificatori di base — Naive Bayes, LDA/QDA, logistica, KNN

### Naive Bayes (formalizzazione, `day5_lavagnate.pdf`)

- **Teorema di Bayes**: la probabilità **congiunta** si fattorizza come
  `Pr(Y,x) = Pr(Y|x)·Pr(x) = Pr(x|Y)·Pr(Y)`, da cui:

  ```
  Pr(Y|x)  ∝  Pr(x|Y) · Pr(Y)
  ```
  (posteriore ∝ verosimiglianza × prior).
- **Assunzione "naive" (ingenua)**: le K feature sono **condizionatamente
  indipendenti dato Y**:

  ```
  Pr(x|Y) = Π_{j=1}^K Pr(x_j | Y)
  ```
  (da cui il nome: si ignora — "ingenuamente" — la dipendenza tra feature).
- **Naive Bayes Gaussiano** (caso con tutte le X numeriche): si assume che,
  condizionatamente a ciascuna classe, x segua una normale multivariata:
  `Pr(x|Y=low) = 𝒩(μ_low, Σ_low)`, `Pr(x|Y=high) = 𝒩(μ_high, Σ_high)`.
- Implementazione: `e1071::naiveBayes(class ~ ., data=x)`.

### Zoo dei classificatori (script `DISD - Classifiers Zoo.R`)

Dati sintetici 2D non linearmente separabili (`mlbench::mlbench.circle`, due
classi a cerchi concentrici), confrontati **visivamente** tramite una funzione
custom `decisionplot()` (fit + contour plot dei confini di decisione su
griglia):

| Metodo | Pacchetto/funzione | Confine di decisione |
|---|---|---|
| **KNN** | `caret::knn3(k=...)` | non parametrico, locale; k=1 → confine molto frastagliato (overfitting); k=10 → più liscio |
| **Naive Bayes** | `e1071::naiveBayes` | assume indipendenza condizionata delle feature dato y |
| **LDA** | `MASS::lda` | **lineare** (assume covarianze uguali tra classi) |
| **QDA** | `MASS::qda` | **quadratico** (covarianze diverse per classe), più flessibile ma rischio overfitting con pochi dati |
| **Logistic Regression** | `glm(family="binomial")` | lineare nel logit, soglia a 0.5 sulla probabilità prevista |
| **Alberi/Ensemble/SVM/NN** | vedi §7-9 | — |

Riferimento: ISLR cap. 4 (logistic regression, LDA, QDA, confronto con KNN).

---

## 7. Alberi decisionali — CART, C4.5/J48, C5.0, ctree

Dataset guida: `GlaucomaMVF` (pacchetto `ipred`) — diagnosi di glaucoma
(binaria: `glaucoma` vs `normal`) da variabili cliniche/oculari.
Script: `materiale/Fine corso/Giorno 3 Prof. Brutti (online)-20260602/CART-screen.R`.

- **Idea generale** (lavagnata giorno 5): gli alberi (versione **CART**)
  eseguono **partizioni ricorsive** dello spazio delle covariate — split
  binari successivi su singole variabili (es. x₁≤5 / x₁>5, poi x₂≤17 / x₂>17,
  ecc.), fino a ottenere regioni rettangolari via via più pure rispetto alla
  classe.
- **CART** (`rpart::rpart`): `rpart.control(xval=100)` per la CV interna usata
  nel pruning cost-complexity.
  - **Pruning**: si guarda la `cptable` (complessità/errore), si sceglie il
    `CP` che minimizza `xerror`, si pota con `prune(tree, cp=...)`.
  - **Instabilità evidenziata esplicitamente dal docente**: alberi diversi
    (fit ripetuti) possono differire — "se il vostro albero è diverso dal
    mio siatene felici": è proprio la motivazione per gli **ensemble**.
  - Visualizzazione con `partykit::as.party(tree)` + `plot()`.
- **C4.5 / J48** (`RWeka::J48`): criterio basato sul guadagno di
  informazione/entropia (CART di default usa Gini). `Weka_control(R=TRUE,
  M=5)` per reduced-error pruning + minimo osservazioni per foglia.
- **C5.0** (`C50::C5.0`): evoluzione di C4.5, supporta **boosting nativo**
  (`trials=N`), probabilità (`type="prob"`), o **regole** (`rules=TRUE`).
  `C5imp()`/`varImp()` per l'importanza delle variabili.
- **Conditional inference trees** (`party::ctree`): split scelti con test di
  significatività statistica invece di puro guadagno di impurità.

### Visualizzazione e overfitting degli alberi (`CART-plot.R`)

Script dedicato (dataset `caret::segmentationData`, segmentazione cellulare):
- `prp()` (pacchetto `rpart.plot`) e `fancyRpartPlot()` (pacchetto `rattle`)
  per visualizzazioni leggibili, alternative a `plot(tree); text(tree)`
  (illeggibile su alberi grandi).
- **Dimostrazione esplicita di overfitting**: un "big tree" fittato con
  `rpart.control(minsplit=20, cp=0)` (nessuna penalità sulla complessità) vs
  un "albero ragionevole" fittato con parametri di default — confronto
  visivo diretto tra un albero enorme/rumoroso e uno compatto.
  - Il prof anticipa anche un caso d'uso avanzato (codice commentato) con
    `RevoScaleR::rxDTree` per alberi su dataset molto grandi.
- **Pruning interattivo**: `prp(tree, snip=TRUE)` permette di potare l'albero
  a mano cliccando sui nodi.

Riferimento: ISLR cap. 8.1 (Tree-Based Methods: costruzione, impurità Gini/
entropia, cost-complexity pruning).

---

## 8. Metodi Ensemble — Bagging, Random Forest, Boosting

### Motivazione e framing bias-variance (`day5_lavagnate.pdf`, "Back in the days")

I singoli alberi sono **instabili** (alta varianza). Gli ensemble aggregano
molti modelli per ottenere un predittore più robusto, ma con **logiche
opposte** a seconda del punto di partenza:

- **Random Forest (RF)**: si parte da **alberi overfittanti** (bias basso,
  varianza alta). L'ensembling (**bagging**) deve **controllare la
  varianza**, sfruttando la proprietà per cui la **media di previsioni tra
  loro indipendenti (o poco correlate) e a bassa distorsione** produce una
  nuova previsione che è ancora a bassa distorsione ma con **variabilità
  ridotta**. Ricetta (Leo Breiman): (1) **Bootstrap** — M ricampionamenti
  𝒟^(1),...,𝒟^(M) di dimensione n, con le osservazioni escluse da ciascun
  campione usate come stima **out-of-bag (OOB)**; (2) **selezione casuale
  delle variabili di split** ad ogni nodo (parametro `mtry`) per
  **decorrelare** ulteriormente gli alberi tra loro.
- **Gradient Boosting (GB)**: si parte da **alberi underfittanti** (bias
  alto, varianza bassa, es. "stumps"). L'ensembling (**boosting**) deve
  **controllare il bias**, costruendo gli alberi **in sequenza**, ciascuno
  che "scava nei residui" del precedente (fit additivo e pesato che corregge
  progressivamente l'errore).
- Nel classico grafico errore-vs-complessità (bias-variance trade-off), RF
  parte da modelli complessi (alta profondità/"depth") e li "media" per
  scendere lungo la curva della varianza; GB parte da modelli semplici e
  sale lungo la curva scavando nel bias.

### Implementazioni pratiche (`CART-screen2.R`)

- **Bagging**:
  - `ipred::bagging(Class ~ ., data, coob=TRUE)` (con stima OOB).
  - **Implementazione a mano** (istruttiva): B campioni bootstrap con
    `rmultinom()`, un albero CART per campione pesato (`update(mod,
    weights=bootsamples[,i])`), predizioni **solo sulle osservazioni non
    incluse nel bootstrap** (le altre a `NA`), media finale → è esattamente
    la stima **OOB**.
- **Random Forest** (`randomForest::randomForest`): bagging + selezione
  casuale delle variabili di split (`mtry`). `importance()`/`varImpPlot()`
  per l'importanza delle variabili (utile anche per confrontare con i
  coefficienti "poco interpretabili" della regressione lineare, §2).
- **Boosting**:
  - `gbm::gbm(distribution="bernoulli")`: gradient boosting.
  - `C50::C5.0(..., trials=10)`: boosting nativo (stile AdaBoost) integrato
    in C5.0.
- Tutti confrontati sullo stesso test set con l'errore di classificazione.

**Nota sull'esercizio del docente**: nello script `day04-05.R` (su
`meatclass`, la versione classificatoria di `meatspec`), il confronto tra
albero singolo e Random Forest è lasciato esplicitamente come esercizio da
completare ("Provate ad adattare una RF? — A voi!"), dopo aver fornito la
soluzione completa per l'albero singolo (vedi §10).

Riferimento: ISLR cap. 8.2 (Bagging, Random Forests, Boosting).

---

## 9. Classificatori a regole

(`materiale/Fine corso/Giorno 3 Prof. Brutti (online)-20260602/CART-screen3.R`)

- **PART** (`RWeka::PART`): estrae un insieme di **regole if-then** da una
  serie di alberi parziali (combina idee di alberi e regole).
- **C5.0 rules** (`C50::C5.0(..., rules=TRUE)`): stessa logica di C5.0 ma
  l'output è un set di regole invece che un albero.
- Vantaggio: **interpretabilità** — ogni predizione è giustificabile con una
  lista esplicita di condizioni, a parità di potere predittivo.

---

## 10. `caret` come framework unificante (workflow completo)

`caret::train()` fornisce un'interfaccia comune a decine di algoritmi
(`method="rpart"/"rpart2"/"pcr"/"pls"/"lm"`, ecc.). Lo script più completo sul
tema (`day04-05.R`) mostra l'intero workflow di model selection **risolto**
per un albero di classificazione su `meatclass`:

1. **`trainControl`**: `method="cv", number=5` (base) oppure, versione più
   ricca, `method="repeatedcv", number=3, repeats=10` con:
   - `savePredictions="final"` (salva le predizioni hold-out del modello
     migliore, per ogni ripetizione della CV)
   - `classProbs=TRUE` (stima le probabilità di classe, non solo l'etichetta)
   - `summaryFunction=twoClassSummary` (metriche di **ROC, sensitivity,
     specificity** invece della sola accuracy — si ricollega a quanto detto
     al §5 sull'insufficienza della perdita 0/1)
2. **`modelLookup(model="rpart2")`**: per scoprire quali iperparametri sono
   tunabili per un dato metodo (utile perché il "nickname" caret di un
   metodo non è sempre ovvio — es. si usa `getModelInfo()`/grep per trovare
   quello giusto).
3. **Tuning**: prima chiamata "di base" (`tree1`, griglia automatica), poi
   griglia esplicita passata a mano (`tree2`, `tuneGrid=data.frame(maxdepth=
   1:20)`).
4. **Diagnosi/lettura del modello**: `tree1$results`, `plot(tree1)` (RMSE/
   Accuracy vs iperparametro), `library(rattle); fancyRpartPlot(tree1$
   finalModel)` per vedere l'albero migliore selezionato.
5. **Importanza delle variabili**: `varImp(tree1)` — per un singolo albero è
   la riduzione di impurità attribuita a ciascuna variabile ad ogni split
   (sommata su tutti gli split); per gli ensemble il calcolo è concettualmente
   meno diretto (vedi documentazione `?varImp`, sezione "Details").
6. **Confronto tra modelli**: `resamples(list(albero1=tree1, albero2=tree2))`
   + `bwplot()`.
7. **Cos'è Kappa?** (Cohen's Kappa) — metrica di accordo osservato vs atteso
   per caso, riportata di default da `caret` insieme all'accuracy (utile
   specialmente con classi sbilanciate, dove l'accuracy da sola è fuorviante).
8. **Valutazione manuale vs automatica**: errore di classificazione a mano
   (`mean(pred==actual)`) e matrice di confusione a mano (`table(pred, true)`)
   **vs** `confusionMatrix(pred, actual)` (che fornisce anche sensitivity,
   specificity, e altre metriche — link fornito dal docente:
   https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
9. **Con `repeatedcv` + `savePredictions`**: `confusionMatrix(tree3$pred$pred,
   tree3$pred$obs)` valuta le performance sulle predizioni hold-out **pooled**
   di tutte le ripetizioni della CV, non solo su un singolo test set.

Questo è il "collante" pratico che lega quasi tutti gli argomenti del corso:
una volta capito bene questo workflow, si applica velocemente a tutti i
modelli (lineari, PCR/PLS, alberi, ensemble).

Documentazione: https://topepo.github.io/caret/

---

## 11. SVM e Kernel Trick

Script `DISD - Kernel Trick.R` e sezione SVM di `DISD - Classifiers Zoo.R`.

- **Idea di base**: trovare l'iperpiano che separa le classi massimizzando il
  margine (distanza dai punti più vicini, i **support vector**).
- **Problema**: molti dati reali non sono linearmente separabili nello spazio
  originale delle feature.
- **Kernel trick — esempio esplicito nello script**: si aggiunge
  esplicitamente una feature `feat = x1² + x2²` (mappa esplicita in 3D). Nel
  nuovo spazio a 3 dimensioni, dati che formavano anelli concentrici in 2D
  diventano **separabili da un piano**. Concretizza l'idea che una
  **mappatura non lineare in dimensione più alta** (esattamente la feature
  map Φ del §1!) può rendere lineare un problema che non lo era.
  Il "trick" vero e proprio nelle SVM è che questa mappatura **non va
  calcolata esplicitamente**: basta una **funzione kernel** che calcoli il
  prodotto scalare nello spazio trasformato senza mai costruirlo.
- **Kernel usati nello Zoo** (`e1071::svm`): `linear`, `radial` (RBF, il più
  usato in pratica per confini non lineari "morbidi"), `polynomial`,
  `sigmoid`.

Riferimento: ISLR cap. 9 (Support Vector Machines).

---

## 12. Reti neurali (cenno)

Nello Zoo dei classificatori, `nnet::nnet(size=k)` addestra una rete
**shallow** (un solo hidden layer) con k neuroni. Aumentando k (1→2→4→10) il
confine di decisione diventa via via più flessibile — stessa logica di
bias-variance vista altrove: più unità nascoste = più capacità di adattarsi a
confini complessi, ma anche più rischio di overfitting.

Riferimento (approfondimento, non coperto in dettaglio nel corso): ISLR
cap. 10 (Deep Learning).

---

## 13. Dataset del corso — riepilogo

| Dataset | Percorso (dopo estrazione) | Tipo di problema | Note |
|---|---|---|---|
| `meatspec` | `Fine corso/Giorno 1 .../day01/day01/meatspec.txt` | Regressione (poi discretizzato in classificazione) | 100 predittori spettrali fortemente collineari; risposta = contenuto di grasso |
| `GlaucomaMVF` | pacchetto `ipred` (`data()`), fold di lavoro in `Fine corso/Giorno 3 .../` | Classificazione binaria | Diagnosi di glaucoma da variabili cliniche/oculari |
| `segmentationData` | pacchetto `caret` (`data()`) | Classificazione | Usato solo per dimostrare visualizzazione/overfitting degli alberi in `CART-plot.R` |
| Circle data | generato con `mlbench::mlbench.circle` | Classificazione binaria sintetica 2D | Classi a cerchi concentrici, non linearmente separabili |
| Kernel trick toy data | definito inline nello script | Classificazione binaria sintetica 2D | 29 punti costruiti a mano per illustrare la feature map |
| `netintru` (NSL-KDD) | `materiale/network/network/netintru.txt` | Classificazione multiclasse (5 classi) | Network intrusion detection; 41 feature (continue/discrete/categoriali) + `Class` (`normal/dos/probe/r2l/u2r`) + `Difficulty Level`; verosimilmente sbilanciato |
| Buzz in social media | `materiale/buzz/buzz/buzzdata.txt` (+ `buzz_info.txt`) | **Classificazione binaria** | 96 feature numeriche = 12 grandezze (NCD, BL, NAD, AI, NAC, ND, CS, AT, NA, ADL, AS(NA), AS(NAC)) misurate su 8 istanti temporali consecutivi → forte struttura di correlazione/ridondanza tra feature (stesso tipo di sfida di `meatspec`, ma in classificazione); risposta binaria `Buzz` (1 = topic diventa "virale", 0 altrimenti) |

Riferimento generale sul confronto tra le famiglie di modelli usate su questi
dataset: ISLR cap. 2 (per capire quando conviene un modello semplice/
interpretabile vs uno complesso/black-box).

---

## Glossario rapido

- **Overfitting**: il modello si adatta troppo al rumore del training set.
- **Bias-variance trade-off**: modelli semplici → alto bias, bassa varianza;
  modelli complessi → basso bias, alta varianza.
- **Cross-validation (K-fold)**: stima onesta dell'errore di generalizzazione
  usando solo dati di training, ripetendo train/validation su K partizioni.
- **Out-of-bag (OOB)**: nelle tecniche bootstrap-based, le osservazioni non
  incluse in un campione bootstrap fungono da validazione "gratuita".
- **Multicollinearità**: forte correlazione lineare tra predittori, che
  destabilizza i coefficienti di un modello lineare — affrontabile con
  selezione di variabili, Ridge/Lasso, o riduzione dimensionale (PCR/PLS).
- **Norma (di θ)**: misura di "grandezza" del vettore dei coefficienti,
  alla base dei metodi penalizzati (Ridge=L2, Lasso=L1).
- **Sparsità**: proprietà di una soluzione con molti coefficienti esattamente
  a zero (tipica del Lasso, non della Ridge).
- **Feature map (Φ)**: trasformazione delle covariate originali in un nuovo
  spazio di rappresentazione; unifica concettualmente PCR/PLS, kernel/SVM e
  reti neurali come istanze dello stesso schema generale f_θ(x)=θᵀΦ(x).
  Impurità (Gini/entropia): misure usate dagli alberi per scegliere lo split
  migliore a ogni nodo.
- **Pruning (potatura)**: rimozione di rami di un albero per ridurne la
  complessità, guidata da CV interna (cost-complexity pruning).
- **Importanza delle variabili** (`varImp`): quanto ciascun predittore
  contribuisce alla capacità predittiva del modello.
- **Perdita 0/1 e surrogate**: la vera metrica di interesse in classificazione
  (errore di classificazione) non è derivabile → si ottimizzano proxy più
  trattabili (es. hinge loss per le SVM, log-loss per la logistica).
- **Oltre l'accuracy**: sensitivity, specificity, ROC/AUC, F1, Kappa di
  Cohen — necessarie quando gli errori non sono tutti equivalenti o le classi
  sono sbilanciate.
