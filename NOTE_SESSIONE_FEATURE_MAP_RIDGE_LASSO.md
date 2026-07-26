# Note di sessione — Feature map, Kernel trick, Day 1-5, Ridge & Lasso, PCA/PCR/PLS, Alberi & Ensemble, Zoo dei classificatori

Appunti da una sessione di Q&A con Claude, con spiegazioni intuitive
(collegate al codice e ai dataset del corso) su alcuni concetti chiave.
Complementari a `SINTESI_CORSO.md` (teoria organizzata per argomento) e
`PIANO_STUDI.md` (percorso di studio).

---

## 1. La feature map (Φ) — spiegazione semplice

**Idea in una frase**: Φ prende i dati e li "guarda da un'altra angolazione"
(li ridisegna in un nuovo spazio) in modo che un problema difficile diventi
facile.

**Analogia**: punti disegnati su un foglio (2D) mescolati a spirale —
impossibile separarli con una linea retta. Se "solleviamo" alcuni punti verso
l'alto in base a quanto sono lontani dal centro (aggiungiamo una terza
dimensione, l'altezza), i punti vicini al centro restano bassi, quelli
lontani salgono. Visti di lato in 3D, ora sono separabili da un piano
orizzontale. Quel "sollevamento" è Φ: non cambia i punti, li *ridescrive*
aggiungendo un'informazione (l'altezza) implicita nei dati originali ma non
esplicita.

**Collegamento allo script `DISD - Kernel Trick.R`**: `x1, x2` sono il foglio
2D dove i punti sono mescolati (nuvola centrale + anello esterno, non
linearmente separabili). `feat = x1²+x2²` è il "sollevare in altezza in base
alla distanza dal centro": i punti centrali restano bassi, quelli esterni
salgono. Il piano che li separa in 3D era la retta impossibile da trovare in
2D.

**Perché "cambio di spazio di rappresentazione"**: lo spazio originale sono
le variabili misurate direttamente (es. x1, x2); il nuovo spazio (dopo Φ) è
la stessa informazione riorganizzata/arricchita in un modo che rende il
problema (separare classi, prevedere y) più facile da risolvere con uno
strumento semplice (retta/piano). Φ non è magia: è una ricetta per calcolare
nuove feature dalle vecchie, scelta per far emergere una struttura
(linearità) nascosta nei dati grezzi.

Nel corso, Φ è la "meta-idea" che unifica regressione lineare (Φ=identità),
PCR/PLS (Φ=componenti principali/PLS), kernel/SVM (Φ=mappatura non lineare,
anche infinito-dimensionale) e reti neurali (Φ=output del hidden layer) —
vedi §1 di `SINTESI_CORSO.md`.

---

## 2. Il kernel trick — spiegazione semplice

**Il problema con "sollevare i punti" a mano**: nell'esempio di `feat =
x1²+x2²` abbiamo dovuto *calcolare davvero* la nuova feature per ogni punto.
Funziona con un solo "sollevamento", ma per dati complicati servirebbero
mappature Φ molto più ricche — decine, migliaia, a volte **infinite**
dimensioni. Calcolare esplicitamente tutte quelle coordinate per ogni punto
sarebbe intrattabile.

**L'osservazione furba**: per trovare la retta/piano separatore, l'algoritmo
(SVM) non ha mai bisogno delle coordinate assolute dei punti nel nuovo
spazio. Ha bisogno solo di un numero per ogni coppia di punti che dica
"quanto sono simili/vicini" in quel nuovo spazio.

**Analogia**: per sapere quanto sono simili due canzoni, un modo è descrivere
ognuna con centinaia di numeri (tempo, tonalità, timbro — le "coordinate"
nello spazio arricchito) e poi confrontare gli elenchi. Un modo più furbo: una
funzione magica che, date due canzoni, restituisce direttamente "quanto sono
simili", **senza mai calcolare i centinaia di numeri intermedi**.

Il **kernel** K(x, x') è questa funzione magica: dà "la somiglianza che
avrebbero x e x' se li avessimo trasformati con Φ", **senza mai calcolare
Φ(x) e Φ(x')** esplicitamente.

**Perché è un "trucco"**: con il kernel si finge di lavorare in uno spazio
arricchito enorme (anche infinito-dimensionale, come nel kernel radiale/RBF)
pagando il costo di lavorare in quello originale (poche variabili). Nello
script, un kernel `polynomial` di grado 2 avrebbe trovato da solo una
separazione equivalente a `feat = x1²+x2²`, senza che nessuno gli dicesse
esplicitamente di aggiungere quella feature.

Riferimento: ISLR cap. 9 (Support Vector Machines); §11 di
`SINTESI_CORSO.md`.

---

## 3. Cosa è stato fatto nel Day 1 (dataset `meatspec`)

**Obiettivo**: prevedere il contenuto di grasso (`fat`) di 215 campioni di
carne a partire da 100 misure spettrali (100 frequenze), e incontrare il
primo grande problema: la **multicollinearità**.

1. **Import ed esplorazione base**: caricamento `meatspec.txt`, controllo
   struttura (`dim`, `str`, `class`), differenza tra selezionare una colonna
   come vettore (`meatspec$fat`) o come data.frame.
2. **Le covariate**: plot del profilo spettrale di una singola unità, poi
   `matplot` di tutti i 215 profili sovrapposti — curve quasi parallele/molto
   simili tra loro.
3. **La risposta (`fat`)**: sintesi numeriche e grafiche; dall'istogramma
   emergono **3 sottopopolazioni** — nota per un possibile approccio di
   classificazione futuro.
4. **Analisi di correlazione** (il cuore della lezione):
   - Tra covariate: altissima (`cor(Xmat)`, `corrplot`/`image`) — conseguenza
     di frequenze vicine quasi identiche.
   - Tra risposta e covariate: pattern forte, con un picco (`max_cor`) a una
     certa frequenza.
5. **Train/test split**: 70/30 casuale con `sample()` e `set.seed()`.
6. **Primo modello: `lm(fat ~ ., data = tr)`** con tutte le 100 variabili:
   - RMSE basso in training, molto più alto in test → **overfitting**.
   - Coefficienti in `summary(mod_all)` totalmente non interpretabili
     (segni/magnitudini assurde) — effetto diretto della multicollinearità.

**Perché conta**: questo esperimento è esattamente il problema che Ridge,
Lasso, PCR e PLS sono costruiti per risolvere (vedi punto 4 sotto).

---

## 4. Ridge e Lasso, partendo dall'esempio del Day 1

**Il problema lasciato in sospeso**: con tutte le 100 frequenze spettrali
(fortemente collineari), `lm` produce coefficienti enormi e instabili — per
inseguire il training set alla perfezione, il modello "gioca" le variabili
una contro l'altra (coefficiente enorme positivo su una frequenza,
compensato da uno enorme negativo su quella adiacente quasi identica). Il
risultato compensa esattamente sul training, ma è instabile sul test → RMSE
di test molto peggiore.

**L'idea di Ridge/Lasso in parole semplici**: se il problema è "i
coefficienti diventano troppo grandi e sregolati", la soluzione è **mettergli
un guinzaglio**. Invece di minimizzare solo l'errore di previsione, si
minimizza l'errore **più una penale per avere coefficienti grandi**. Il
modello può ancora usare tutte le 100 variabili, ma ogni coefficiente grande
"costa caro", quindi non può più permettersi la cancellazione a compensazione
che causava l'overfitting.

**Il parametro λ** (guinzaglio):
- λ = 0 → nessun guinzaglio → `lm` classico, coefficienti liberi di
  esplodere.
- λ grande → guinzaglio stretto → coefficienti schiacciati verso zero,
  modello più prudente.
- si sceglie per cross-validation.

**Ridge vs Lasso — la differenza è come si misura "quanto sono grandi" i
coefficienti**:
- **Ridge** (penalità L2, somma dei quadrati): tutti i coefficienti si
  restringono verso zero, ma nessuno diventa esattamente 0 — soluzione
  "densa". Utile su `meatspec` se si pensa che l'informazione sia distribuita
  su tutto lo spettro: Ridge condivide il peso tra le 100 frequenze invece di
  farne esplodere alcune.
- **Lasso** (penalità L1, somma dei valori assoluti): alcuni coefficienti
  vengono azzerati esattamente — selezione automatica delle variabili. Su
  `meatspec`: "di queste 100 frequenze quasi ridondanti, ne tengo solo alcune
  (es. vicino al picco di correlazione `max_cor`) e butto via le altre".

**Aspettativa**: sia Ridge sia Lasso, a differenza del `lm` del Day 1,
dovrebbero dare un RMSE di test molto più vicino a quello di training,
risolvendo l'overfitting osservato.

Riferimento: ISLR cap. 6.2 (Ridge Regression and the Lasso); §3 di
`SINTESI_CORSO.md`; formalizzazione matematica completa in
`materiale/Fine corso/Lavagnate-20260602/day4_lavagnate.pdf`.

---

## 5. Cosa è stato fatto nel Day 2 (script `day02.R`, dataset `meatspec`)

Il Day 2 riparte esattamente da dove si era fermato il Day 1: `lm(fat ~ .,
data = tr)` con tutte le 100 frequenze overfitta (RMSE train basso, RMSE test
molto peggiore, coefficienti non interpretabili per multicollinearità). Il
Day 2 esplora due famiglie di soluzioni, entrambe alternative "storiche" a
Ridge/Lasso (visti concettualmente solo alla lavagna, §4 sopra):

1. **Tentativo 1 — selezione per correlazione**: idea semplice, "butto via le
   variabili troppo correlate tra loro". Con `caret::findCorrelation(Xcor_tr,
   cutoff = ...)` si ottengono indici di colonne da eliminare; il cutoff è un
   vero e proprio **iperparametro di tuning** (con `cutoff = 0.99` restano
   quasi tutte le 100 variabili — la corr. media assoluta è .98 — con
   `cutoff` più permissivo ne restano pochissime).
2. **Come scegliere il cutoff in modo "predittivo"**: qui entra la
   **k-fold cross-validation**, implementata a mano nello script (non con
   `caret::train`, per vedere l'ingranaggio):
   - si etichettano le osservazioni di training in `K=3` fold con
     `sample(rep(1:K, length = nrow(tr)))`;
   - per ogni configurazione di variabili (`idxA`, `idxB`, `idxC`, nessuna
     esclusione), si fa un ciclo su `k = 1..K`: si adatta `lm` sui fold
     `!= k`, si predice sul fold `k` (mai visto in quel training), si calcola
     l'RMSE; la media sui K fold è la **K-CV estimate** per quella
     configurazione;
   - il grafico finale (`plot(nvar, err_tr, ...)` + `lines(nvar, KCV_vec,
     ...)`) mostra il classico **bias-variance trade-off**: l'errore di
     training scende monotonamente aumentando il numero di variabili (sempre
     più ottimistico, non fidarsene mai da solo), mentre il CV-score ha un
     andamento a "U" — troppe poche variabili → troppo bias, troppe variabili
     (con n piccolo per fold) → troppa varianza/instabilità.
   - **Nota tecnica importante lasciata nello script**: con K=3 fold da ~52
     osservazioni, il modello con tutte le 100 variabili viene stimato con
     appena ~104 gradi di libertà effettivi → numericamente instabile, CV-score
     "ridicolmente alto" per quella configurazione. È un avvertimento pratico:
     la CV non è magica, va usata con giudizio quando p è vicino a n (o p>n).
3. **Tentativo 2 — PCR e PLS** (vedi §6-7 sotto), stavolta con `caret::train`
   e CV automatica (`trainControl(method="cv", number=3)`), e infine
   confronto di tutti i modelli (`lm` ridotto, PCR, PLS) con
   `resamples()`/`bwplot()`.

Riferimento: ISLR cap. 5 (Cross-Validation) per il punto 2; cap. 6.3
(Dimension Reduction Methods) per PCR/PLS; §4 di `SINTESI_CORSO.md`.

---

## 6. PCA spiegata con l'esempio del cavallo (`horse-PCA.pdf`)

**Il documento** `materiale/Fine corso/Giorno 2 Prof. Brutti-20260602/horse-PCA.pdf`
mostra la nuvola di punti 3D che forma la sagoma di un cavallo, e costruisce
la PCA passo-passo:

- **Pagina 1**: un punto `x` del cavallo viene proiettato (ortogonalmente) su
  una direzione `d1` che passa per il centro `x0` della nuvola. Il segmento
  rosso `d1ᵀx` è "l'ombra" di `x` su quella direzione — un solo numero al
  posto delle 3 coordinate originali. `d1` non è scelta a caso: è la
  direzione lungo cui, proiettando *tutti* i punti del cavallo, l'ombra
  varia di più (massima varianza) — è la **prima componente principale**.
- **Pagina 2**: si aggiunge una seconda direzione `d2`, ortogonale a `d1`,
  che cattura la varianza "rimasta" (quella non già spiegata da `d1`).
  Proiettando su entrambe (`d1`, `d2`) si ottiene un'ombra 2D del cavallo 3D
  (visibile nel riquadro a sinistra): si perde informazione (la profondità),
  ma si conserva la maggior parte della "forma" riconoscibile.
- **Pagina 3**: con una terza direzione `d3` (ortogonale alle prime due) si
  ricostruiscono tutte e 3 le dimensioni originali — `d1, d2, d3` sono
  semplicemente un **nuovo sistema di assi**, ruotato rispetto a quello
  originale e allineato con le direzioni di massima variabilità dei dati.

**Perché è la stessa idea di Φ (§1)**: la PCA è una feature map Φ molto
particolare — lineare, e scelta *senza guardare la risposta y*, solo dalla
struttura di correlazione delle X. Su `meatspec`, invece di un cavallo in 3D
abbiamo una nuvola in 100 dimensioni (le frequenze spettrali, fortissimamente
correlate tra loro): la PCA trova poche direzioni (`d1, d2, ...`, cioè le
**componenti principali**, `princomp(tr[,1:100])` nello script) lungo cui la
variabilità spettrale si concentra quasi per intero — proprio come poche
direzioni bastano a "riconoscere" il cavallo pur perdendo una dimensione.

**Attenzione (nota già nello script)**: le direzioni di massima varianza nelle
X non sono necessariamente quelle più utili per prevedere `fat` — varianza
spiegata nelle covariate ≠ potere predittivo sulla risposta. Questo è
precisamente il limite della PCA/PCR che PLS cerca di correggere (§7).

---

## 7. PCR e PLS, partendo dall'esempio del Day 2

**PCR (Principal Component Regression)**: prima si calcolano le componenti
principali delle 100 frequenze (come nel cavallo, §6 — *senza* guardare
`fat`), poi si fa una regressione lineare di `fat` sulle prime K componenti
invece che sulle 100 variabili originali. K è un iperparametro di tuning
(`tuneLength = 30` nello script, scelto per CV via `caret::train(...,
method="pcr")`) — esattamente lo stesso schema logico del cutoff di
`findCorrelation` in §5, ma qui il "riassunto" delle 100 variabili non è
"tenerne un sottoinsieme" bensì "combinarle linearmente in poche direzioni".
Risultato nello script: RMSE di test migliore di tutti i modelli precedenti
(`lm` completo, `lm` ridotto per correlazione).

**PLS (Partial Least Squares)**: stessa struttura (regressione lineare su
poche combinazioni lineari delle X), ma le combinazioni non sono scelte solo
guardando la varianza delle X (come in PCR) — sono scelte guardando anche
**quanto covariano con la risposta `fat`**. È la correzione al problema
notato in §6: la PCA/PCR può "sprecare" le prime componenti su direzioni ad
alta varianza ma poco informative per y; PLS cerca direttamente le direzioni
più predittive. Nello script, PLS raggiunge prestazioni comparabili o
migliori della PCR con **meno componenti** — coerente con l'idea che, essendo
supervisionato, "spende" ogni componente in modo più efficiente.

**Perché sono ancora un caso di Φ (§1)**: PCR ha Φ(x) = prime K componenti
principali; PLS ha Φ(x) = prime K componenti PLS. È lo stesso schema
concettuale di Ridge/Lasso (§4) applicato al *tipo* di problema (troppe
variabili collineari) ma con una strategia diversa: invece di "tenere tutte
le variabili ma penalizzare i coefficienti grandi" (Ridge/Lasso), PCR/PLS
"riducono prima il numero di variabili trasformandole" e poi fanno `lm`
classico su quelle poche.

**Confronto finale nello script** (`resamples()` + `bwplot(resamps, metric =
"RMSE")`): mette a confronto, sugli stessi fold di CV, PCR, PLS e `lm`
ridotto per correlazione (`modB`, §5) — un modo onesto di scegliere tra
strategie diverse usando la stessa metrica e lo stesso ricampionamento,
invece di confrontare numeri ottenuti in modi diversi.

Riferimento: ISLR cap. 6.3 (Dimension Reduction Methods: PCR, PLS); §4 di
`SINTESI_CORSO.md`.

---

## 8. Cosa è stato fatto nel Day 3 (script `CART-screen.R`, `CART-screen2.R`,
`CART-screen3.R`, `CART-plot.R`, dataset `GlaucomaMVF`)

Il Day 3 cambia dataset e problema — non più `meatspec` (regressione), ma
`GlaucomaMVF` (pacchetto `ipred`): diagnosi binaria `glaucoma` vs `normal` da
variabili cliniche/oculari — e cambia famiglia di modelli: dagli alberi
lineari (lm, Ridge/Lasso, PCR/PLS) agli **alberi decisionali (CART)** e ai
loro **ensemble**. La parte formale/completa è già in `SINTESI_CORSO.md` §7
(alberi: CART/rpart, C4.5-J48, C5.0, ctree, pruning, `CART-plot.R`), §8
(bagging/random forest/boosting) e §9 (classificatori a regole, PART/C5.0
rules) — qui aggiungo solo il livello "spiegazione in parole semplici" che
collega questi metodi a quanto già visto (§1-7).

---

## 9. Perché un albero singolo è "capriccioso" — l'instabilità dei CART

**Cosa nota esplicitamente il docente in `CART-screen.R`**: rifittando lo
stesso albero (`rpart(Class ~ ., data = training)`) con condizioni iniziali
leggermente diverse, l'albero risultante può essere diverso da quello di un
altro studente sugli stessi dati — "se il vostro albero è diverso dal mio
siatene felici!".

**Perché succede**: un albero costruisce i suoi split in modo **greedy e
gerarchico** — sceglie la variabile e la soglia che separano meglio *in
quel momento*, poi ripete sui sotto-gruppi. Se per un pugno di osservazioni
in più o in meno il primo split cambia (es. tra due variabili quasi
altrettanto informative), **tutto l'albero sotto quel nodo cambia**, perché
ogni split successivo dipende da quello precedente. È il motivo per cui gli
alberi sono descritti come modelli ad **alta varianza**: piccole
perturbazioni nei dati di training → risultati anche molto diversi.

**Collegamento a quanto già visto**: è lo stesso tipo di instabilità
osservato nel Day 1 con `lm` sulle 100 frequenze collineari (§3) — lì la
causa era la multicollinearità, qui è la natura gerarchica/greedy degli
split — ma l'effetto (un modello "nervoso", sensibile ai dati) è concettualmente
lo stesso, e la soluzione ha la stessa struttura logica: **non fidarsi di un
singolo fit, ma combinarne molti** (ensemble) o **penalizzare/limitare la
complessità** (pruning, §11).

---

## 10. Bagging, Random Forest e Boosting — la stessa idea di "combinare tanti
pareri", con due logiche opposte

**Analogia**: un albero singolo è come chiedere una diagnosi a un solo medico
un po' "umorale" — bravo in media, ma la risposta può cambiare parecchio da
un giorno all'altro (alta varianza). Due modi diversi per ottenere una
diagnosi più affidabile chiedendo a più medici:

- **Bagging / Random Forest — "tanti pareri indipendenti, poi si fa la
  media"**: si fanno crescere **alberi grandi, volutamente overfittanti**
  (bias basso, varianza alta — come il `mod_all` con 100 variabili nel Day 1,
  ma qui l'"overfitting" è nella profondità dell'albero) su tanti campioni
  bootstrap diversi (`ipred::bagging`, o "a mano" con `rmultinom()` +
  `update(mod, weights=...)` in `CART-screen2.R`), e poi si **media** il loro
  parere. Mediare pareri **indipendenti** (o poco correlati) riduce la
  varianza senza aumentare il bias — è la stessa logica per cui la media di
  tanti termometri leggermente imprecisi ma indipendenti è più affidabile di
  un termometro solo. La **random forest** (`randomForest::randomForest`)
  aggiunge un trucco in più: ad ogni split, ogni albero può scegliere solo tra
  un **sottoinsieme casuale di variabili** (`mtry`) — questo "decorrela"
  ulteriormente i medici tra loro (impedisce che copino tutti la stessa
  variabile più forte), rendendo la media ancora più efficace.
- **Boosting — "una squadra di specialisti in sequenza, ognuno corregge
  l'errore del precedente"**: si parte da alberi **piccoli e deliberatamente
  poco potenti** ("stumps": bias alto, varianza bassa — il medico junior che
  sbaglia spesso ma in modo consistente), e si costruiscono **in sequenza**:
  ogni nuovo albero si concentra su ciò che i precedenti hanno sbagliato
  (`gbm::gbm`, oppure il boosting nativo di `C50::C5.0(..., trials=10)`, in
  stile AdaBoost). Qui non si media per ridurre la varianza (è già bassa); si
  **somma** in modo pesato per ridurre progressivamente il **bias**.

**In una frase**: bagging/RF partono "in alto" (modelli complessi, alta
varianza) e scendono mediando; il boosting parte "in basso" (modelli
semplici, alto bias) e sale sommando correzioni successive. Sono le due facce
opposte dello stesso trade-off bias-varianza incontrato la prima volta nel
Day 2 (§5, il grafico train-error vs CV-score).

Riferimento: ISLR cap. 8.2; §8 di `SINTESI_CORSO.md`; formalizzazione
completa (bootstrap, OOB, `mtry`, framing bias-variance) in
`materiale/Fine corso/Lavagnate-20260602/day5_lavagnate.pdf`.

---

## 11. Pruning — lo stesso "guinzaglio" di λ (Ridge/Lasso) e del cutoff (Day 2),
ma per gli alberi

**Il problema**: un albero fatto crescere senza limiti (`rpart.control(minsplit
= 20, cp = 0)` in `CART-plot.R`) si adatta perfettamente al training set —
uno split per ogni piccola irregolarità dei dati — ed è quindi il tipico
overfitting già visto altrove nel corso (Day 1 con 100 variabili, Day 2 con
troppe componenti/troppe poche esclusioni).

**Il parametro `cp` (complexity parameter)**: gioca lo stesso ruolo di λ in
Ridge/Lasso (§4) e del `cutoff` in `findCorrelation` (§5) — un numero che
penalizza la complessità (qui: il numero di split) e va scelto per
cross-validation, non a occhio. `rpart` calcola automaticamente una
`cptable` con l'errore di CV interno (`xerror`) per diversi valori di `cp`;
si sceglie il `cp` che minimizza `xerror` (`which.min(...)`) e si pota
l'albero con `prune(tree, cp = cp_ottimo)`.

**Perché è di nuovo lo stesso schema**: in tutti e quattro i casi (Ridge/Lasso,
cutoff di correlazione, K di PCR/PLS, `cp` degli alberi) la ricetta è
identica — (1) un solo numero controlla quanto il modello può essere
complesso, (2) si costruisce una griglia di valori candidati, (3) si stima
l'errore predittivo per ciascuno **via cross-validation sul solo training
set**, (4) si sceglie il valore che minimizza quella stima. Cambia solo
*cosa* viene reso più semplice (coefficienti, variabili, componenti, split).

Riferimento: ISLR cap. 8.1 (cost-complexity pruning); §7 di
`SINTESI_CORSO.md`.

---

## 12. Cosa è stato fatto nel Day 4-5 (script `day04-05.R`, `DISD - Classifiers
Zoo.R`; il kernel trick di `DISD - Kernel Trick.R` è già in §1-2)

Il Day 4-5 chiude il cerchio tra regressione e classificazione, riusando
proprio `meatspec`: la risposta continua `fat` viene **discretizzata** in due
classi ("low"/"high"), prima a occhio con una soglia arbitraria (`cut(jnk,
breaks = c(min, 22, max))`), poi in modo più sistematico con `kmeans(jnk,
centers = 2)` (si etichettano i due cluster guardando i loro centri) — nasce
così `meatclass`, la "versione categoriale" di `meatspec` già anticipata
nel Day 1 (§3, le 3 sottopopolazioni nell'istogramma di `fat`). Dopo uno
split stratificato (`createDataPartition`, classi moderatamente
sbilanciate), lo script fa da traccia guidata: adatta un albero singolo con
`caret::train(method="rpart2")` — prima con tuning di default, poi passando
una griglia esplicita (`tuneGrid = data.frame(maxdepth = 1:20)`) — confronta
due alberi con `resamples()`/`bwplot()`, introduce `confusionMatrix()` e la
CV ripetuta con metriche dedicate alla classificazione
(`trainControl(summaryFunction = twoClassSummary)`), e lascia **come
esercizio esplicito** l'adattare una Random Forest sullo stesso problema
("Provate ad adattare una RF? — A voi!") confrontandola con l'albero singolo
tramite `varImp()` — lo stesso confronto albero-vs-ensemble già visto nel
Day 3 (§9-10), qui su un problema di classificazione invece che su
`GlaucomaMVF`.

`DISD - Classifiers Zoo.R` fa invece un esperimento diverso e molto
istruttivo: uno **stesso identico dataset sintetico 2D** non linearmente
separabile (`mlbench.circle`, due classi a cerchi concentrici — la stessa
struttura "nuvola centrale + anello esterno" già incontrata nel kernel trick,
§1) viene affrontato con **una dozzina di classificatori diversi** (KNN,
Naive Bayes, LDA, QDA, logistica, alberi, C5.0, random forest, SVM con vari
kernel, reti neurali con diverse dimensioni), visualizzando il confine di
decisione di ciascuno con la funzione custom `decisionplot()`.

Riferimento: ISLR cap. 4; §5-6 e §10-11 di `SINTESI_CORSO.md` (dettaglio
completo di ogni classificatore).

---

## 13. Perché "accuratezza" non basta — spiegazione intuitiva

**Il problema**: contare semplicemente quante previsioni sono sbagliate
(la **perdita 0/1**, `mean(pred != actual)`, usata negli script fin dal primo
albero su `meatclass`) tratta **tutti gli errori allo stesso modo**. Ma un
falso positivo e un falso negativo spesso non sono equivalenti.

**Analogia**: un test diagnostico per una malattia grave. Sbagliare dicendo
"sano" a un malato (**falso negativo**) può essere molto più grave di
sbagliare dicendo "malato" a un sano (**falso positivo**, che nel peggiore
dei casi porta solo ad accertamenti ulteriori). Un modello con il 95% di
accuratezza che sbaglia sistematicamente proprio sui malati veri (magari
perché la classe "malato" è rara) è **inutile in pratica**, anche se il
numero sembra ottimo.

**Da qui la necessità di guardare oltre l'accuratezza** — esattamente ciò
che `caret::confusionMatrix()` restituisce: sensitivity (quanti malati veri
vengono identificati), specificity (quanti sani veri vengono identificati
correttamente), e con `twoClassSummary`/curve ROC anche una vista che non
dipende da una singola soglia di decisione (0.5) ma da come sensitivity e
specificity si scambiano al variare della soglia.

**Cos'è il Kappa** (di Cohen, citato nello script): è l'accuratezza
"corretta per il caso" — confronta quanto il modello ci azzecca in più
rispetto a un classificatore che indovinasse a caso rispettando le stesse
proporzioni di classe. Utile perché, con classi sbilanciate (come qui, "low"
vs "high" moderatamente sbilanciate), un'accuratezza alta può nascondere un
modello che si limita a prevedere quasi sempre la classe maggioritaria.

Riferimento: ISLR cap. 4 (Classification, oltre l'accuracy); §5 di
`SINTESI_CORSO.md`; formalizzazione (perdita 0/1, hinge loss come
"surrogato") in `materiale/Fine corso/Lavagnate-20260602/day5_lavagnate.pdf`.

---

## 14. Naive Bayes spiegato in parole semplici

**L'idea di base (teorema di Bayes)**: invece di modellare direttamente "qual
è la classe più probabile dati questi valori delle variabili", si ribalta il
problema — si modella "quanto sono tipici questi valori delle variabili
*per ciascuna classe*" (`Pr(x|Y)`, la verosimiglianza) e si combina con
quanto è frequente ciascuna classe in generale (`Pr(Y)`, il prior):

```
Pr(Y|x)  ∝  Pr(x|Y) · Pr(Y)
```

**Analogia**: per capire se un frutto sconosciuto è una mela o un'arancia,
invece di avere una regola diretta "se colore=arancione e forma=tonda allora
arancia", si ragiona così: "tra tutte le arance che ho mai visto, quante
erano arancioni e tonde? tra tutte le mele, quante erano arancioni e
tonde?" — e si sceglie la classe per cui la combinazione osservata è più
tipica, pesando anche quanto sia comune quella classe in generale (se le mele
sono il 90% dei frutti nel cesto, serve un'evidenza più forte per convincersi
che è un'arancia).

**Perché "naive" (ingenua)**: calcolare `Pr(x|Y)` per tutte le combinazioni
di feature insieme sarebbe complicatissimo (servirebbe sapere come colore,
forma, peso... covariano tra loro dentro ciascuna classe). Naive Bayes fa una
scorciatoia drastica: **assume che le feature siano indipendenti tra loro,
una volta nota la classe** — `Pr(x|Y) = Π_j Pr(x_j|Y)`, cioè valuta ogni
variabile per conto suo e moltiplica i contributi, ignorando ("ingenuamente")
eventuali correlazioni tra colore e forma. È un'assunzione quasi sempre
falsa in pratica, ma spesso funziona bene lo stesso — è veloce, richiede poche
osservazioni per stimare i parametri, ed è una base di confronto naturale per
gli altri classificatori dello zoo (§15).

**Nello script**: `e1071::naiveBayes(class ~ ., data = x)` sul dataset a
cerchi concentrici — il confine di decisione che ne risulta è tipicamente più
"morbido"/meno preciso di SVM o alberi su questo problema, proprio perché
l'indipendenza condizionata è un'assunzione forte e qui palesemente non vera
(le due variabili `x, y` sono legate insieme dalla distanza dal centro).

Riferimento: ISLR cap. 4.4 (Naive Bayes); §6 di `SINTESI_CORSO.md`;
formalizzazione completa in
`materiale/Fine corso/Lavagnate-20260602/day5_lavagnate.pdf`.

---

## 15. Lo zoo dei classificatori — stesso problema, confini diversi: ancora
bias-variance

**L'esperimento in `DISD - Classifiers Zoo.R`** è pensato apposta per
mostrare, sullo stesso identico dataset (i cerchi concentrici non linearmente
separabili), quanto sia **diversa la forma del confine di decisione** che
ciascuna famiglia di modelli riesce a disegnare — è la stessa domanda del
Day 1 (che modello si adatta al problema?) ma vista attraverso un grafico
invece che un numero (RMSE):

- **KNN**: con `k=1` il confine è frastagliatissimo — ogni punto di training
  "vota" da solo, il modello **overfitta** violentemente (varianza altissima,
  bias basso). Con `k=10` il confine si smussa: ogni previsione media il
  voto dei 10 vicini più prossimi, più bias ma meno varianza. Stesso
  trade-off del Day 2 (numero di componenti in PCR/PLS, §5-7) e del Day 3
  (`cp` degli alberi, §11), qui col parametro `k`.
- **LDA vs QDA**: LDA assume che le classi abbiano la **stessa forma di
  covarianza** e traccia un confine **lineare** — troppo rigido per i cerchi
  concentrici (che richiedono un confine curvo), quindi **underfitta**. QDA
  permette covarianze diverse per classe e quindi confini **quadratici**
  (curvi) — più adatto qui, ma con il rischio di overfittare se i dati per
  classe sono pochi (più parametri da stimare). È lo stesso spettro
  bias-variance di prima, applicato alla *forma* del confine invece che al
  numero di variabili/componenti.
- **Naive Bayes**: confine intermedio, limitato dall'assunzione di
  indipendenza (§14).
- **Regressione logistica**: come LDA, confine lineare (nel logit) — stesso
  limite sui cerchi concentrici, a meno di aggiungere a mano feature non
  lineari (esattamente il ruolo di Φ, §1!).
- **Alberi/ensemble/SVM/reti neurali**: confini via via più flessibili
  (rettangolari e a gradini per gli alberi, curvi e "globali" per SVM con
  kernel radiale, arbitrariamente complessi per le reti con più unità
  nascoste) — la stessa idea di Φ (kernel trick, §2) o di composizione
  gerarchica (reti) che permette di "piegare" un confine lineare in uno
  curvo senza dover scegliere a mano la trasformazione giusta.

**In una frase**: lo zoo dei classificatori non è un elenco di alternative
scollegate — è la stessa storia raccontata nelle sezioni precedenti (un
parametro/un'assunzione che controlla quanto il modello può essere flessibile,
e va scelto guardando l'errore di previsione, non quello di training) vista
"in pianta" invece che in un grafico errore-vs-complessità.

Riferimento: ISLR cap. 4 (LDA/QDA/logistica/KNN) e cap. 9 (SVM); §6 e §11 di
`SINTESI_CORSO.md`.

---

## 16. SVM: cosa vuol dire "massimizzare il margine" — spiegazione intuitiva

**Il problema che risolve**: dati due gruppi di punti separabili (almeno
approssimativamente) da una retta/piano, ci sono **infinite** rette che li
separano correttamente. Quale scegliere?

**Analogia**: immagina di dover tracciare una strada dritta tra due gruppi di
case (rosse e blu) senza toccarne nessuna. Potresti tracciarla vicinissima
alle case rosse, o vicinissima alle blu, o esattamente a metà — tutte
"funzionano" nel senso che separano i due gruppi. La SVM sceglie la strada
**più larga possibile**: quella che lascia il massimo margine di sicurezza da
entrambi i lati. Intuitivamente, una strada larga è più "robusta": un nuovo
punto leggermente fuori posto ha meno probabilità di finire dal lato
sbagliato rispetto a una strada strettissima incollata a un gruppo.

**I "support vector"**: una volta tracciata la strada più larga possibile,
solo le case **esattamente sul bordo della strada** contano per definirla —
sono i punti di supporto. Tutte le altre case, più lontane dal confine,
potrebbero sparire senza cambiare la strada di un millimetro. È il motivo del
nome: il modello finale dipende solo da un sottoinsieme (spesso piccolo) delle
osservazioni, non da tutte.

**Collegamento al kernel trick (§1-2)**: la "strada dritta" funziona solo se
i due gruppi sono ragionevolmente separabili in linea retta. Sui cerchi
concentrici dello Zoo dei classificatori (§15), nessuna retta funziona — ed è
esattamente lì che entra il kernel: si finge di lavorare in uno spazio
trasformato (§1-2) dove una retta *esiste*, e nello Zoo lo si vede
concretamente confrontando `kernel="linear"` (fallisce sui cerchi) con
`"radial"`/`"polynomial"` (si adattano al confine curvo) — il kernel radiale
in particolare è il più usato in pratica proprio perché produce confini curvi
"morbidi" senza dover scegliere a mano quale trasformazione esplicita usare.

Riferimento: ISLR cap. 9 (Support Vector Machines, in particolare la nozione
di margine e support vector prima di introdurre i kernel); §11 di
`SINTESI_CORSO.md`.

---

## 17. Reti neurali (cenno): comporre tante funzioni semplici — spiegazione
intuitiva

**L'idea di base**: una rete neurale shallow (un solo hidden layer, come
`nnet::nnet(size = k)` nello Zoo) non è altro che tanti piccoli
"classificatori lineari" (i neuroni del hidden layer, ciascuno una
combinazione lineare degli input passata attraverso una funzione non
lineare) le cui uscite vengono a loro volta combinate linearmente per dare
la previsione finale.

**Analogia**: pensa a un comitato di k "giudici junior", ognuno dei quali
guarda gli stessi dati in ingresso ma si concentra su una propria
combinazione/prospettiva particolare (un giudice potrebbe reagire forte solo
quando `x1` è grande, un altro solo quando `x1` e `x2` sono entrambi
piccoli, ecc.), e restituisce un giudizio semplice (sì/no, o un numero). Un
giudice "capo" (l'ultimo strato) ascolta tutti i giudici junior e combina i
loro pareri per la decisione finale. Nessun singolo giudice junior sa
separare bene i cerchi concentrici — ma la combinazione delle loro opinioni
parziali sì.

**Perché è di nuovo un caso di Φ (§1)**: i k neuroni del hidden layer sono
esattamente una feature map **appresa dai dati** invece che scelta a mano
(come `feat = x1²+x2²` nel kernel trick, §1) o implicita (come nel kernel
delle SVM, §2/§16) — la rete impara da sola quali "nuove coordinate"
rendono il problema linearmente risolvibile nell'ultimo strato.

**Il parametro `size` (k, numero di neuroni)**: gioca lo stesso ruolo di
tutti gli altri iperparametri di complessità già visti (§11, tabella)
— nello Zoo, aumentando k (1→2→4→10) il confine di decisione diventa via via
più flessibile: con pochi neuroni la rete non riesce a "circondare" bene i
cerchi concentrici (underfitting), con più neuroni il confine si adatta
meglio ma cresce anche il rischio di overfitting (specialmente su pochi dati
e senza early stopping/regolarizzazione). Stesso trade-off bias-variance,
stesso schema logico di scelta (griglia di `size` + CV), diversa "manopola".

Riferimento: ISLR cap. 10 (Deep Learning — approfondimento, il corso ne dà
solo un cenno pratico); §12 di `SINTESI_CORSO.md`.
