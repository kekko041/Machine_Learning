# Riassunto — An Introduction to Statistical Learning (2nd ed.)

Riassunto per lo studio di `materiale/ISLRv2_corrected_June_2023-1.pdf`
(James, Witten, Hastie, Tibshirani — 2ª edizione, "Corrected Printing: June
21, 2023"). 13 capitoli + introduzione, ~600 pagine.

Questo documento riassume **l'intero libro**, capitolo per capitolo, ma con
**profondità diversa**: i capitoli 2-6, 8, 9 sono quelli su cui si basa
direttamente il corso (vedi `SINTESI_CORSO.md`) e sono trattati più a fondo,
con formule; i capitoli 7, 10-13 (non coperti dal corso, o solo accennati)
sono riassunti in modo più sintetico, utile comunque per una visione
d'insieme o per approfondimenti facoltativi.

Per collegare ogni argomento al codice R del corso, vedi `SINTESI_CORSO.md` e
`PIANO_STUDI.md`. Sito ufficiale del libro (PDF gratuito, dataset, slide):
https://www.statlearning.com/

---

## Capitolo 1 — Introduction

Capitolo introduttivo, non tecnico. Presenta tre dataset guida usati nel
libro (`Wage`, `Smarket`/Stock Market, `NCI60`) e la distinzione di fondo tra
**apprendimento supervisionato** (regressione, classificazione) e
**non supervisionato** (clustering, PCA). Storia del campo: Statistical
Learning come evoluzione/ramo della statistica classica, con forte overlap col
Machine Learning.

---

## Capitolo 2 — Statistical Learning

Il capitolo **fondativo**: quasi tutto il resto del libro è un'elaborazione
dei concetti qui introdotti.

### 2.1 Cos'è lo Statistical Learning

- Modello generale: **Y = f(X) + ε**, dove f è la relazione sistematica
  (sconosciuta) tra i predittori X e la risposta Y, ε è errore casuale
  (irriducibile) indipendente da X con media 0.
- **Perché stimare f?** Due motivi, spesso in tensione tra loro:
  - **Prediction**: prevedere Y per nuove osservazioni; qui f può restare
    una "black box" — conta solo l'accuratezza.
  - **Inference**: capire **come** Y dipende da X (quali predittori sono
    associati, con che segno/forza, se la relazione è lineare) — qui serve
    interpretabilità.
- **Come si stima f?**
  - **Metodi parametrici**: si assume una forma funzionale (es. lineare) e
    si stimano un numero finito di parametri. Più semplici, ma rischio di
    scegliere una forma troppo lontana dalla vera f.
  - **Metodi non parametrici**: nessuna assunzione esplicita sulla forma di
    f, più flessibili, ma richiedono molti più dati e sono più a rischio
    overfitting.
- **Trade-off accuratezza-interpretabilità**: modelli più flessibili
  (random forest, SVM, reti) tipicamente più accurati in previsione ma meno
  interpretabili di modelli semplici (regressione lineare, lasso).
- **Supervised vs Unsupervised**: con risposta osservata vs senza risposta
  (solo struttura nei dati X).
- **Regression vs Classification**: risposta quantitativa vs qualitativa
  (categoriale).

### 2.2 Valutare l'accuratezza del modello

- **Regressione — Mean Squared Error (MSE)**:
  ```
  MSE = (1/n) Σᵢ (yᵢ − f̂(xᵢ))²
  ```
  Interessa il **test MSE** (su dati non usati per stimare f̂), non il
  training MSE, che è sistematicamente troppo ottimistico (specialmente per
  modelli flessibili → overfitting).
- **Decomposizione bias-variance** del test MSE atteso in un punto x₀:
  ```
  E[(y₀ − f̂(x₀))²] = Var(f̂(x₀)) + [Bias(f̂(x₀))]² + Var(ε)
  ```
  - **Var(f̂(x₀))**: quanto cambierebbe f̂ se stimata su un training set
    diverso — modelli più flessibili hanno varianza più alta.
  - **Bias(f̂(x₀))**: errore introdotto approssimando un problema reale
    (magari molto complesso) con un modello più semplice — modelli meno
    flessibili hanno bias più alto.
  - **Var(ε)**: errore irriducibile, limite inferiore invalicabile del test
    MSE, indipendente dal modello scelto.
  - Al crescere della flessibilità del modello: **bias scende, varianza
    sale** — il test MSE ha tipicamente una forma a **U**, con un minimo al
    compromesso ottimale (training MSE invece scende sempre monotonamente).
- **Classificazione**: la metrica analoga è il **tasso di errore**:
  ```
  (1/n) Σᵢ 1( yᵢ ≠ ŷᵢ )
  ```
  - **Classificatore di Bayes**: assegna ogni osservazione alla classe più
    probabile dato x, cioè `argmax_j Pr(Y=j | X=x)`. È il classificatore
    con tasso di errore più basso **possibile** (errore di Bayes = errore
    irriducibile in classificazione), ma richiede di conoscere la vera
    distribuzione condizionata (in pratica sconosciuta) → si stima con KNN,
    regressione logistica, LDA/QDA, ecc.
  - **KNN** come esempio di stima non parametrica di Pr(Y|X): al crescere di
    K il confine di decisione diventa più liscio (meno flessibile, più
    bias, meno varianza); K piccolo → più flessibile (meno bias, più
    varianza). Stessa logica bias-variance della regressione.

### 2.3-2.4 Lab ed esercizi

Introduzione a R (non centrale per la sintesi teorica).

**Concetti chiave da portare a casa**: la decomposizione bias-variance è **la**
lente attraverso cui leggere tutto il resto del libro — ogni tecnica successiva
(regolarizzazione, alberi, ensemble, SVM) è, in sostanza, un modo diverso di
gestire questo compromesso.

---

## Capitolo 3 — Linear Regression

### Regressione semplice (una covariata)

```
Y ≈ β₀ + β₁X
```
Stima con **minimi quadrati (OLS)**: minimizza RSS = Σᵢ(yᵢ − β̂₀ − β̂₁xᵢ)².
Formule chiuse per β̂₀, β̂₁. **Standard error** dei coefficienti → intervalli
di confidenza e **test d'ipotesi** (t-test, H₀: β₁=0, "non c'è relazione tra
X e Y"). **RSE** (Residual Standard Error) e **R²** (proporzione di
varianza spiegata) per valutare il fit complessivo del modello.

### Regressione multipla

```
Y ≈ β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
```
- **F-test** (H₀: β₁=...=βₚ=0) per verificare se **almeno un** predittore è
  utile — necessario perché con tanti predittori, per puro caso, alcuni
  p-value individuali risulteranno piccoli anche sotto H₀.
- **Selezione delle variabili** (anticipa il cap. 6): forward/backward/
  mixed stepwise selection.
- **Model fit**: R² cresce sempre aggiungendo variabili (anche inutili) →
  serve **R² aggiustato** o le metriche del cap. 6 (Cp, AIC, BIC) per un
  confronto onesto tra modelli di dimensione diversa.

### Altre considerazioni

- **Predittori qualitativi**: variabili dummy/indicatrici.
- **Estensioni**: termini di **interazione** (X₁×X₂, "effetto sinergico" —
  l'effetto di X₁ su Y dipende dal livello di X₂), **non linearità**
  tramite trasformazioni polinomiali (X², X³, ...).
- **Problemi potenziali**: non linearità della vera relazione, correlazione
  tra i residui, eteroschedasticità (varianza non costante dei residui),
  outlier, punti leva (*high leverage points*), **multicollinearità**
  (predittori fortemente correlati tra loro → coefficienti instabili,
  standard error gonfiati, difficile isolare l'effetto individuale di
  ciascun predittore — **esattamente il problema centrale del dataset
  `meatspec` nel corso**). Diagnostica: **VIF** (Variance Inflation Factor).

### Regressione lineare vs KNN

KNN batte la regressione lineare quando la vera f è molto non lineare, **ma
solo se n è grande rispetto a p**: al crescere di p (dimensionalità), le
prestazioni di KNN degradano rapidamente (*curse of dimensionality* — con
molte dimensioni, i "vicini" più prossimi non sono più davvero vicini). È il
motivo per cui, con p grande, spesso un modello parametrico anche
imperfetto batte un modello non parametrico flessibile.

Riferimento diretto al corso: §2 di `SINTESI_CORSO.md`.

---

## Capitolo 4 — Classification

### Perché non regressione lineare per y categoriale?

Con più di 2 classi non ordinate, non c'è modo sensato di codificarle
numericamente per una regressione lineare; anche con 2 classi (codificate
0/1), la regressione lineare può produrre previsioni fuori da [0,1], non
interpretabili come probabilità.

### Regressione logistica

```
p(X) = Pr(Y=1|X) = e^(β₀+β₁X) / (1 + e^(β₀+β₁X))
```
equivalente a linearità nel **log-odds** (logit):
```
log( p(X) / (1−p(X)) ) = β₀ + β₁X
```
Stima via **massima verosimiglianza** (non minimi quadrati). Interpretazione
dei coefficienti: β₁ è la variazione nel log-odds per unità di X (non
un effetto lineare diretto su p(X), a differenza della regressione lineare).
**Regressione logistica multipla** (più predittori) e **multinomiale** (più
di 2 classi, una classe di riferimento/baseline).

### Modelli generativi (LDA, QDA, Naive Bayes)

Approccio alternativo: modellare `Pr(X|Y=k)` per ciascuna classe e la
`Pr(Y=k)` a priori, poi ottenere `Pr(Y=k|X)` via **Teorema di Bayes**:
```
Pr(Y=k|X=x) = π_k · f_k(x) / Σⱼ π_j · f_j(x)
```
dove π_k = Pr(Y=k) (prior) e f_k(x) = densità di X data la classe k.

- **LDA (Linear Discriminant Analysis)**: assume f_k gaussiana con **la
  stessa matrice di covarianza Σ per tutte le classi** → confine di
  decisione **lineare**. Utile quando le classi sono ben separate (la
  logistica può essere instabile in quel caso) o con n piccolo e X
  approssimativamente normali.
- **QDA (Quadratic Discriminant Analysis)**: come LDA ma con **matrice di
  covarianza Σ_k diversa per ciascuna classe** → confine di decisione
  **quadratico**. Più flessibile di LDA (meno bias) ma più parametri da
  stimare (più varianza) — preferibile con n grande o quando l'assunzione
  di covarianze uguali è chiaramente violata.
- **Naive Bayes**: assume che, **data la classe**, i predittori siano
  **condizionatamente indipendenti**:
  ```
  f_k(x) = Π_{j=1}^p f_kj(x_j)
  ```
  Semplificazione "ingenua" ma spesso efficace, specialmente con p grande
  (dove stimare la piena struttura di covarianza per LDA/QDA sarebbe
  impraticabile). Coincide esattamente con quanto formalizzato alla
  lavagna nel corso (`SINTESI_CORSO.md` §6).

### Confronto tra metodi di classificazione

- **Confronto analitico**: LDA è un caso particolare di regressione
  logistica quando i predittori sono gaussiani con covarianza comune (stessa
  forma funzionale del log-odds); Naive Bayes è un caso particolare di GAM
  (cap. 7); QDA sta "a metà" tra LDA/Naive Bayes e metodi più flessibili
  come KNN.
- **Confronto empirico**: nessun metodo domina sempre — dipende dalla vera
  forma del confine di decisione (lineare → LDA/logistica; moderatamente non
  lineare → QDA; molto non lineare/irregolare → KNN, ma serve n grande
  rispetto a p).

### Modelli Lineari Generalizzati (GLM)

Generalizzazione che include regressione lineare, logistica, e **Poisson
regression** (per risposte di conteggio) come casi particolari, unificati
da: una distribuzione della risposta nella famiglia esponenziale + una
**funzione di link** che collega E[Y|X] al predittore lineare.

Riferimento diretto al corso: §5-6 di `SINTESI_CORSO.md`.

---

## Capitolo 5 — Resampling Methods

### Cross-Validation

- **Validation set approach**: singolo split train/validation — semplice
  ma con alta variabilità (dipende molto da quale split si è scelto) e
  tende a sovrastimare il test error (si allena su meno dati).
- **LOOCV (Leave-One-Out CV)**: n fit, ciascuno lasciando fuori una sola
  osservazione. Bassa distorsione (usa quasi tutti i dati per allenare) ma
  costoso computazionalmente (tranne per OLS/ridge, dove esiste una
  scorciatoia in forma chiusa) e con **stime tra loro molto correlate**
  (alta varianza dell'errore CV complessivo, perché i training set si
  sovrappongono quasi interamente).
- **k-fold CV**: compromesso pratico standard (tipicamente k=5 o k=10).
  Meno costoso di LOOCV, con un miglior compromesso bias-variance della
  *stima dell'errore stesso*: LOOCV ha meno bias ma più varianza; k-fold
  con k moderato ha lievemente più bias ma molta meno varianza.
- **CV in classificazione**: stessa idea, ma con il tasso di errore di
  classificazione al posto del MSE.

Esattamente lo schema K-fold formalizzato alla lavagna nel corso (§1 di
`SINTESI_CORSO.md`).

### Bootstrap

Tecnica generale per quantificare l'incertezza di **qualsiasi** stimatore
(anche quando una formula analitica per il suo standard error non esiste o è
complicata): si ricampiona **con reinserimento** dal dataset originale (B
campioni bootstrap di dimensione n), si ricalcola la statistica di interesse
su ciascun campione, e si usa la variabilità tra le B stime come stima
dell'incertezza. Base teorica del **bagging** (cap. 8) e delle stime
**out-of-bag (OOB)**.

Riferimento diretto al corso: §1 e §8 di `SINTESI_CORSO.md`.

---

## Capitolo 6 — Linear Model Selection and Regularization

Tre famiglie di alternative alla regressione con OLS su tutte le variabili,
motivate dagli stessi problemi (troppi predittori, multicollinearità,
overfitting, interpretabilità) visti nel corso su `meatspec`.

### 6.1 Subset Selection

- **Best Subset Selection**: prova **tutti** i 2^p sottoinsiemi possibili di
  predittori, sceglie il migliore per ciascuna dimensione d, poi confronta i
  modelli di dimensione diversa con una metrica che penalizza la
  complessità (sotto). Computazionalmente intrattabile per p grande.
- **Stepwise Selection** (forward, backward, mixed): euristiche molto più
  economiche che esplorano solo un sottoinsieme dei modelli possibili
  (forward parte da zero variabili e aggiunge una alla volta quella che
  migliora di più il fit; backward parte da tutte e rimuove; mixed combina
  entrambe le mosse). Non garantiscono di trovare il modello ottimo globale,
  ma sono fattibili anche con p grande (backward richiede n>p, forward no).
- **Scegliere il modello ottimo** (tra dimensioni diverse): serve una
  stima **onesta** del test error, non il training RSS/R² (che migliorano
  sempre aggiungendo variabili). Due strade:
  1. **Metriche corrette analiticamente** per la complessità del modello:
     ```
     Cp  = (1/n)( RSS + 2·d·σ̂² )                              (6.2)
     AIC = (1/n)( RSS + 2·d·σ̂² )   (∝ Cp per modelli gaussiani/OLS)
     BIC = (1/n)( RSS + log(n)·d·σ̂² )                          (6.3)
     ```
     dove d = numero di predittori, σ̂² = stima della varianza dell'errore
     (dal modello completo). **BIC penalizza di più** la complessità quando
     n>7 (poiché log(n)>2), scegliendo modelli tipicamente più piccoli di
     Cp/AIC. **R² aggiustato** è un'alternativa nello stesso spirito (cresce
     solo se una nuova variabile migliora il fit più di quanto ci si
     aspetterebbe per caso).
  2. **Validation/Cross-Validation diretta**: si stima il test error
     direttamente su dati tenuti da parte, senza bisogno di stimare σ̂² o
     contare i gradi di libertà — approccio più moderno e generalmente
     preferito, perché richiede meno assunzioni.

### 6.2 Shrinkage Methods (Ridge e Lasso)

Alternativa alla selezione discreta di sottoinsiemi: adattare **tutte** le p
variabili ma **restringere (shrink)** i coefficienti verso zero.

- **Ridge Regression**:
  ```
  β̂^ridge = argmin_β  Σᵢ(yᵢ − β₀ − Σⱼβⱼxᵢⱼ)²  +  λ·Σⱼβⱼ²
  ```
  Penalità **L2**. λ≥0 è il parametro di tuning (**λ=0** → OLS ordinario;
  **λ→∞** → tutti i coefficienti verso 0). Nota: la penalità **non include
  l'intercetta** β₀ (che rappresenta solo la media di Y, non va ristretta).
  **Importante**: la scala delle variabili conta (a differenza di OLS,
  Ridge non è invariante per riscalamento) → **standardizzare sempre i
  predittori** prima di adattare Ridge/Lasso. Ridge non produce mai
  coefficienti esattamente zero (a meno di casi degeneri) → tutte le
  variabili restano nel modello finale, solo "ristrette".
- **The Lasso**:
  ```
  β̂^lasso = argmin_β  Σᵢ(yᵢ − β₀ − Σⱼβⱼxᵢⱼ)²  +  λ·Σⱼ|βⱼ|
  ```
  Penalità **L1**. A differenza della Ridge, produce soluzioni **sparse**:
  per λ sufficientemente grande, alcuni coefficienti sono **esattamente
  zero** → il Lasso esegue **selezione automatica delle variabili**, dando
  modelli più interpretabili. Spiegazione geometrica standard (identica al
  disegno del docente nella lavagnata del corso): la regione di vincolo
  ‖β‖₁≤s ha "spigoli" sugli assi, dove è più probabile che l'ottimo cada,
  mentre la regione ‖β‖₂≤s (Ridge) è una sfera liscia senza spigoli.
- **Scegliere λ**: sempre per **cross-validation** — si prova una griglia di
  valori di λ, si calcola l'errore di CV per ciascuno, si sceglie il λ che
  lo minimizza (o, per un modello più parsimonioso, la regola "one-standard-
  error": il λ più grande il cui errore CV è entro 1 SE dal minimo).
- **Ridge vs Lasso, quando preferire cosa**: se ci si aspetta che **molti**
  predittori abbiano un effetto piccolo ma non nullo → **Ridge** di solito
  vince in accuratezza predittiva. Se ci si aspetta che **pochi** predittori
  siano davvero rilevanti (il resto ha effetto ~0) → **Lasso** di solito
  vince, ed è anche più interpretabile.

Riferimento diretto al corso: §3 di `SINTESI_CORSO.md` (dove la derivazione
è presentata in notazione matriciale, con soluzione chiusa della Ridge
`θ̂^R(λ)=(XᵀX+λI)⁻¹Xᵀy` — coerente con quanto qui descritto).

### 6.3 Dimension Reduction Methods (PCR, PLS)

Terza famiglia: invece di selezionare o restringere le p variabili originali,
**trasformarle** in M<p combinazioni lineari (componenti) Z₁,...,Z_M, e fare
regressione su queste.

- **PCR (Principal Component Regression)**: le componenti sono le **componenti
  principali** di X (cap. 12), scelte in modo **non supervisionato** per
  massimizzare la varianza catturata in X, **senza guardare Y**. Rischio:
  le direzioni di massima varianza in X non sono necessariamente le più
  predittive per Y. M (numero di componenti) è scelto via CV.
- **PLS (Partial Least Squares)**: alternativa **supervisionata** — le
  componenti sono costruite usando anche Y (tramite le covarianze
  univariate tra ciascun predittore e la risposta come pesi iniziali), per
  massimizzare la covarianza con Y oltre alla varianza in X.

### 6.4 High-Dimensional Considerations

Quando **p > n** (o p vicino a n): OLS classico è mal definito/instabile (fit
perfetto ma inutile in test — *curse of dimensionality* in regressione);
servono necessariamente metodi con regolarizzazione (Ridge/Lasso) o
riduzione dimensionale. Attenzione anche nell'**interpretazione**: con p>n
ci sono infinite soluzioni con RSS training = 0, quindi trovare un buon
sottoinsieme di predittori non significa aver trovato "il" sottoinsieme
causalmente rilevante — serve validazione indipendente (nuovo dataset) prima
di trarre conclusioni forti.

Riferimento diretto al corso: §3-4 di `SINTESI_CORSO.md`.

---

## Capitolo 7 — Moving Beyond Linearity

*(Non trattato esplicitamente nel corso — utile come approfondimento facoltativo
sul tema "linearità vs non linearità", a metà tra regressione lineare/PCR-PLS
e alberi/SVM.)*

- **Polynomial regression**: aggiungere X², X³, ... come predittori — resta
  un modello lineare **nei parametri**, ma non linare in X.
- **Step functions**: discretizzare X in intervalli (bin) e adattare una
  costante diversa per bin — evita di imporre una forma globale a f.
- **Basis functions**: framework generale che include entrambi i casi sopra:
  `f(X) = β₀ + β₁b₁(X) + ... + β_Kb_K(X)` per funzioni di base b_k
  predefinite (esattamente la feature map Φ vista alla lavagna nel corso!).
- **Regression/Smoothing splines**: polinomi a tratti (*piecewise*) uniti
  con vincoli di continuità/derivabilità ai nodi (*knots*); le smoothing
  splines usano un parametro di penalità λ sulla "rugosità" (curvatura) della
  curva invece di scegliere esplicitamente il numero di nodi — stesso spirito
  di Ridge/Lasso ma per funzioni invece che coefficienti.
- **Local regression**: fit di un modello semplice separatamente in ogni
  intorno locale di X (con pesi decrescenti con la distanza).
- **GAM (Generalized Additive Models)**: estendono la regressione (lineare o
  logistica) permettendo una funzione non lineare **separata** per ciascun
  predittore: `Y = β₀ + f₁(X₁) + f₂(X₂) + ... + fₚ(Xₚ) + ε` — flessibili
  ma restano additivi (niente interazioni automatiche tra predittori).

---

## Capitolo 8 — Tree-Based Methods

### 8.1 Alberi di base

- **Alberi di regressione**: **partizione ricorsiva e binaria** dello spazio
  dei predittori in regioni rettangolari R₁,...,R_J (algoritmo *greedy*
  **top-down** e **recursive binary splitting**: ad ogni passo si sceglie
  la coppia predittore/soglia che minimizza il RSS risultante, senza
  guardare al futuro — da qui "greedy", non garantisce l'albero ottimo
  globale). La previsione per ogni regione è la **media** delle y di
  training in quella regione.
  - **Cost-complexity pruning**: si cresce prima un albero grande (fino a
    un criterio di stop minimo, es. poche osservazioni per foglia), poi lo
    si pota massimizzando `RSS + α·|T|` (|T| = numero di foglie), con α
    scelto per CV — analogo del Lasso applicato alla dimensione dell'albero.
- **Alberi di classificazione**: stessa logica, ma la previsione per regione
  è la **classe più frequente**; il criterio di split usa una misura di
  **impurità** invece del RSS:
  - **Gini index**: `G = Σ_k p̂_mk(1−p̂_mk)` — misura la "purezza" di un nodo
    (piccolo se un nodo è dominato da una sola classe).
  - **Entropy**: `D = −Σ_k p̂_mk·log(p̂_mk)` — comportamento numerico simile
    al Gini index, entrambi preferiti al semplice tasso di errore di
    classificazione per la costruzione dell'albero (più sensibili alla
    purezza dei nodi).
- **Alberi vs modelli lineari**: gli alberi sono superiori quando la vera
  relazione è fortemente non lineare/con interazioni complesse tra
  predittori "a soglia"; i modelli lineari vincono quando la vera relazione
  è approssimativamente lineare.
- **Vantaggi**: facili da spiegare/interpretare (anche più della regressione
  lineare secondo molti), gestiscono nativamente predittori categoriali
  senza bisogno di dummy, si prestano a una visualizzazione intuitiva.
  **Svantaggi**: accuratezza predittiva tipicamente inferiore ad altri
  metodi, **molto instabili** (piccole variazioni nei dati training →
  struttura dell'albero molto diversa — alta varianza) — motivazione diretta
  per gli ensemble.

### 8.2 Bagging, Random Forest, Boosting, BART

- **Bagging (Bootstrap Aggregating)**: si generano B dataset bootstrap dal
  training set, si adatta un albero (tipicamente cresciuto in profondità,
  senza pruning — basso bias, alta varianza) su ciascuno, e si mediano le
  previsioni (per regressione) o si vota la classe più frequente (per
  classificazione). La media di B stime **poco correlate** ha varianza
  ridotta di un fattore ~1/B rispetto a una singola stima, a parità di bias
  — è il principio statistico di base che giustifica il bagging. Stima
  dell'errore **out-of-bag (OOB)**: ogni albero è allenato solo su ~2/3
  delle osservazioni (in media); le rimanenti ~1/3 ("out-of-bag" per
  quell'albero) possono essere usate come validazione gratuita, senza
  bisogno di CV separata.
- **Random Forest**: come il bagging, ma ad ogni split si considera solo un
  **sottoinsieme casuale di m predittori** (tipicamente m≈√p per
  classificazione, m≈p/3 per regressione) invece di tutti i p. Motivazione:
  se esiste un predittore molto forte, il bagging tenderebbe a usarlo quasi
  sempre come primo split in ogni albero → alberi molto **correlati** tra
  loro → mediare non riduce la varianza quanto potrebbe. Limitando le
  variabili disponibili ad ogni split si **decorrelano** gli alberi,
  migliorando la riduzione di varianza dell'ensemble.
- **Boosting**: costruzione **sequenziale** (non su bootstrap indipendenti):
  ogni nuovo albero (tipicamente piccolo, "stump"-like, con pochi split) è
  adattato sui **residui** del modello corrente (non sulla risposta
  originale), e la sua previsione viene aggiunta al modello con un fattore
  di **shrinkage** (learning rate) λ piccolo. Iperparametri chiave: **B**
  (numero di alberi — a differenza del bagging/RF, un B troppo grande **può**
  causare overfitting, va scelto per CV), **λ** (shrinkage/learning rate,
  tipicamente piccolo, 0.01-0.001), **d** (profondità di ciascun albero,
  spesso d=1, gli "stump", sufficiente se non ci sono interazioni forti tra
  predittori).
- **BART (Bayesian Additive Regression Trees)**: combina idee di bagging e
  boosting — mantiene un insieme di alberi che vengono aggiornati
  iterativamente (in stile MCMC/Bayesiano) invece che una sola volta.
- **Riepilogo**: bagging/RF = alberi **paralleli e indipendenti** (via
  bootstrap) mediati per **ridurre la varianza**; boosting = alberi
  **sequenziali e dipendenti** (via residui) per **ridurre il bias**;
  entrambi risolvono l'instabilità (alta varianza) del singolo albero, ma
  con logiche opposte — esattamente il framing "Back in the days" della
  lavagnata del corso.

Riferimento diretto al corso: §7-8 di `SINTESI_CORSO.md`.

---

## Capitolo 9 — Support Vector Machines

- **Iperpiano separatore**: in p dimensioni, un iperpiano è definito da
  `β₀+β₁X₁+...+βₚXₚ=0`; se le classi sono perfettamente separabili, esistono
  infiniti iperpiani separatori.
- **Maximal Margin Classifier**: tra tutti gli iperpiani separatori, sceglie
  quello che **massimizza il margine** (la distanza minima dai punti di
  training più vicini, i **support vector** — solo questi punti determinano
  la posizione dell'iperpiano). Esiste solo se le classi sono perfettamente
  separabili (caso raro/fragile in pratica: molto sensibile a singole
  osservazioni).
- **Support Vector Classifier**: generalizzazione con un **margine morbido**
  (*soft margin*) che tollera alcune osservazioni dalla parte sbagliata del
  margine (o anche dell'iperpiano), controllato da un parametro di tuning
  **C** (budget di violazioni tollerate): **C piccolo** → margine stretto,
  meno violazioni tollerate, meno bias/più varianza; **C grande** → margine
  largo, più violazioni tollerate, più bias/meno varianza (scelto per CV,
  stesso trade-off bias-varianza di sempre).
- **Support Vector Machine (kernel non lineare)**: quando il confine di
  decisione naturale non è lineare, si amplia lo spazio dei predittori con
  **funzioni kernel**, che calcolano il prodotto scalare in uno spazio
  trasformato (anche a dimensione molto alta o infinita) **senza calcolarlo
  esplicitamente** — il "kernel trick" propriamente detto. Kernel comuni:
  **polinomiale** `K(xᵢ,xᵢ′)=(1+Σxᵢⱼxᵢ′ⱼ)^d`, **radiale/RBF**
  `K(xᵢ,xᵢ′)=exp(−γΣ(xᵢⱼ−xᵢ′ⱼ)²)` (γ controlla quanto "localmente" il
  confine può curvare — γ grande → confine molto flessibile/rischio
  overfitting). Esattamente l'esempio esplicito con `feat=x1²+x2²` visto nel
  corso è un caso particolare (kernel polinomiale di grado 2 costruito a
  mano invece che tramite la funzione kernel implicita).
- **SVM con più di 2 classi**: **One-vs-One** (un classificatore per ogni
  coppia di classi, vince chi ottiene più "voti") o **One-vs-All** (un
  classificatore per ciascuna classe contro tutte le altre, si sceglie la
  classe col punteggio più alto).
- **Relazione con la regressione logistica**: la SVM può essere vista come
  un problema di minimizzazione regolarizzata `Σᵢ max(0, 1−yᵢf(xᵢ)) + λ‖β‖²`
  con la **hinge loss** al posto della log-loss della logistica — le due
  loss sono simili, e infatti le prestazioni predittive delle due tecniche
  sono spesso comparabili; SVM è preferita quando le classi sono ben
  separate, logistica quando servono anche stime di probabilità.

Riferimento diretto al corso: §11 di `SINTESI_CORSO.md`.

---

## Capitolo 10 — Deep Learning

*(Il corso tocca solo superficialmente le reti neurali shallow — `nnet` nello
Zoo dei classificatori. Qui un riassunto essenziale per contestualizzare.)*

- **Rete neurale a singolo strato (single layer)**: input X → **K unità
  nascoste** `A_k = g(w_{k0} + Σⱼw_{kj}Xⱼ)` (g = funzione di attivazione non
  lineare, es. **ReLU** `g(z)=max(0,z)` o sigmoide) → output
  `f(X)=β₀+Σ_kβ_kA_k`. Il numero K di unità nascoste è l'analogo del numero
  di componenti in PCR/PLS o della complessità di un albero: più K → più
  flessibilità (meno bias, più varianza).
- **Reti multilayer**: più hidden layer in sequenza, ciascuno che trasforma
  l'output del precedente — permette di rappresentare funzioni via via più
  complesse (rappresentazioni gerarchiche delle feature).
- **CNN (Convolutional Neural Networks)**: specializzate per immagini —
  **strati convoluzionali** (filtri/kernel che rilevano pattern locali,
  condividendo pesi su tutta l'immagine) e **strati di pooling**
  (sottocampionamento che riduce la dimensionalità mantenendo le feature
  rilevanti).
- **RNN (Recurrent Neural Networks)**: specializzate per dati sequenziali
  (testo, serie temporali) — l'output ad ogni step dipende anche da uno
  stato "nascosto" che riassume la sequenza vista finora.
- **Fitting**: tramite **backpropagation** (discesa del gradiente + regola
  della catena per calcolare i gradienti rispetto a tutti i pesi), spesso
  in versione **stocastica (SGD)** su mini-batch; regolarizzazione tramite
  **weight decay** (= Ridge sui pesi) o **dropout** (si "spengono" a
  caso alcune unità durante il training, per evitare co-adattamento).
- **Double Descent**: fenomeno per cui, in modelli molto sovraparametrizzati
  (numero di parametri ≫ n, tipico del deep learning), il test error, dopo
  essere salito nel regime di "interpolazione perfetta" (training error=0),
  **torna a scendere** aumentando ulteriormente la complessità — anticipa
  quanto accennato a lezione nel corso a proposito di p>n.

Riferimento diretto al corso: §12 di `SINTESI_CORSO.md`.

---

## Capitolo 11 — Survival Analysis and Censored Data

*(Non trattato nel corso — riassunto sintetico.)*

- Studia il **tempo al verificarsi di un evento** (sopravvivenza, guasto,
  ecc.), con il problema tipico della **censura** (per molte unità
  l'evento non si osserva ancora entro la fine dello studio — non è un dato
  mancante qualunque, ha struttura specifica).
- **Curva di sopravvivenza di Kaplan-Meier**: stima non parametrica di
  `S(t)=Pr(T>t)` che tiene conto correttamente della censura.
- **Log-rank test**: confronta le curve di sopravvivenza tra due gruppi.
- **Modello di Cox a rischi proporzionali**: regressione sulla **funzione di
  rischio istantaneo** (hazard), senza dover specificare la forma della
  baseline hazard — l'analogo, per dati di sopravvivenza, di una
  regressione lineare/logistica; si estende con Ridge/Lasso allo stesso
  modo del cap. 6.

---

## Capitolo 12 — Unsupervised Learning

- **PCA (Principal Components Analysis)**: trova le direzioni (componenti
  principali) di massima varianza in X, in sequenza e ortogonali tra loro —
  la stessa intuizione geometrica illustrata col modello 3D di cavallo
  nella lavagnata del corso (§4 di `SINTESI_CORSO.md`). Usata per
  visualizzazione (prime 2-3 componenti), riduzione dimensionale
  (pre-processing per PCR, cap. 6), o come tecnica esplorativa a sé.
  **Proportion of Variance Explained (PVE)**: quanta varianza totale è
  catturata da ciascuna componente — tipicamente visualizzata con uno
  **scree plot** per decidere quante componenti mantenere ("gomito" della
  curva).
- **Matrix Completion**: uso della PCA (via SVD) per **imputare valori
  mancanti** in una matrice di dati (idea alla base dei sistemi di
  raccomandazione).
- **Clustering**:
  - **K-means**: partiziona le osservazioni in K gruppi minimizzando la
    varianza intra-cluster (somma delle distanze quadratiche dal centroide
    del proprio cluster); K va scelto a priori, algoritmo iterativo
    (assegna punti al centroide più vicino → ricalcola centroidi → ripeti),
    sensibile all'inizializzazione (si usano più partenze casuali,
    `nstart`, esattamente come nello script del corso che discretizza
    `fat` con `kmeans(centers=2, nstart=10)`).
  - **Hierarchical clustering**: costruisce un **dendrogramma** (albero di
    fusioni successive, bottom-up/agglomerativo) senza dover fissare K a
    priori; il numero di cluster si sceglie **tagliando** il dendrogramma a
    un'altezza scelta. **Linkage** (come misurare la distanza tra due
    cluster già formati): *complete* (max distanza), *single* (min
    distanza, tende a produrre cluster "a catena"), *average*, *centroid*.
  - **Problemi pratici comuni**: scala delle variabili (spesso serve
    standardizzare prima), scelta della metrica di distanza, scelta di
    K/altezza di taglio — nessuna risposta univoca, richiede giudizio e
    validazione di dominio.

---

## Capitolo 13 — Multiple Testing

*(Non trattato nel corso — riassunto sintetico, utile se si eseguono molti
test d'ipotesi in parallelo, es. su feature genomiche o A/B test multipli.)*

- **Problema**: testando m ipotesi contemporaneamente con soglia α=0.05
  ciascuna, la probabilità di **almeno un falso positivo** cresce rapidamente
  con m (anche se tutte le H₀ sono vere).
- **Family-Wise Error Rate (FWER)**: probabilità di **almeno un** falso
  positivo tra gli m test. **Correzione di Bonferroni**: usa soglia α/m per
  singolo test → controllo conservativo del FWER, ma perde molta potenza
  statistica quando m è grande.
- **False Discovery Rate (FDR)**: proporzione **attesa** di falsi positivi
  tra le scoperte dichiarate significative (non tra tutti i test) — criterio
  meno conservativo, più adatto quando si accetta qualche falso positivo
  pur di trovare più veri effetti. **Procedura di Benjamini-Hochberg** per
  controllare l'FDR a un livello desiderato.
- **Approcci basati su ri-campionamento**: stimare p-value/FDR direttamente
  dai dati (permutazioni) quando le assunzioni distribuzionali classiche non
  reggono.

---

## Come usare questo riassunto con Claude

- Per collegare un capitolo al codice del corso: *"nel cap. 8 di ISLR si
  parla di cost-complexity pruning — dove lo vedo applicato negli script del
  corso?"* (risposta: `CART-screen.R`, `cptable`/`prune`, vedi
  `SINTESI_CORSO.md` §7).
- Per approfondire un argomento non coperto dal corso (cap. 7, 10-13):
  chiedi pure spiegazioni aggiuntive o esempi — sono comunque utili per
  la comprensione generale del campo anche se non sono materia d'esame.
- Per esercitarti: alla fine di ogni capitolo del PDF ci sono un **Lab** (in
  R, riproducibile) e una sezione **Exercises** — buona palestra pratica in
  aggiunta agli script del corso.
