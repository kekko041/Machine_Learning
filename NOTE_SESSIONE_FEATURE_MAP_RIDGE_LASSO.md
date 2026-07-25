# Note di sessione — Feature map, Kernel trick, Day 1, Ridge & Lasso

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
