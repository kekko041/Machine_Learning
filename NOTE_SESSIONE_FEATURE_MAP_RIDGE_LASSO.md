# Note di sessione — Feature map, Kernel trick, Day 1-2, Ridge & Lasso, PCA/PCR/PLS

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
