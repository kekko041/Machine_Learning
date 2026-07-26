# Spiegazione Day 1 - Data Mining e Machine Learning

Questo documento raccoglie tutti i chiarimenti relativi al file e al codice della prima lezione (Day 1).

## 1. Attenzione: L'input della selezione è un data.frame!
Nel codice troviamo questa istruzione: `plot(meatspec[1,1:10])` accompagnata dal commento *"Attenzione! L'input della selezione è un data.frame!"*.
In R, se si estrae una singola colonna, questa spesso viene convertita in un vettore, ma estraendo più colonne (come in questo caso, 10), il risultato rimane un `data.frame` (di dimensioni 1 riga x 10 colonne). 
La funzione `plot()` ha un comportamento che dipende dalla natura dell'oggetto che le viene passato:
* Passando un **vettore numerico**, `plot()` disegna il classico grafico a dispersione o a linee.
* Passando un **data.frame**, `plot()` cerca di generare una matrice di scatterplot per tutte le variabili (pairs plot), risultando in un grafico illeggibile per una singola osservazione.
Per risolvere il problema ed effettuare il plot dell'andamento dello spettro, si forza la conversione con `as.numeric(meatspec[1, 1:10])`.

## 2. Obiettivo e Preparazione dei Predittori (Covariate)
L'obiettivo (GOAL) è prevedere la variabile `fat` (contenuto di grasso) partendo dai dati spettrometrici.
Il codice estrae le feature nel modo seguente:
```r
Xmat <- as.matrix(meatspec[,1:100])
```
* Scegliendo la **via 1**, vengono usati 100 predittori distinti. La matrice estratta ha tutte le righe e le prime 100 colonne (quelle relative allo spettro).
* L'istruzione `as.matrix()` converte il data.frame in una vera e propria matrice numerica. Gran parte degli algoritmi per modelli multivariati e funzioni grafiche (es. `matplot`) richiedono come input delle matrici per eseguire operazioni algebriche rapide ed efficienti.

## 3. Preparazione della Variabile Risposta (Target)
```r
yvec <- meatspec$fat
summary(yvec) # utile per relativizzare la scala degli errori in previsione
```
* **L'estrazione**: L'operatore `$` estrae la singola colonna `fat` sotto forma di vettore (salvata come `yvec`). 
* **Il Summary**: la funzione `summary()` restituisce statistiche descrittive (minimo, quartili, media, massimo). È vitale usarla per *relativizzare la scala degli errori*: se non si conosce l'ordine di grandezza e la variabilità della risposta `y`, non è possibile capire se un errore (es. RMSE pari a 3.5) sia accettabile o disastroso.

## 4. Panoramica Generale del Flusso (day01.html)
Il documento `day01` illustra il workflow di base:
1. **Esplorazione visiva**: grafici per lo spettro di un pezzo e per tutti i pezzi. Valutazione della distribuzione di `fat` (forse 3 sottopopolazioni).
2. **Analisi di correlazione**: le variabili X sono fortissimamente correlate tra loro (multicollinearità). Viene cercata la singola X più correlata alla variabile Y.
3. **Modello Base Iniziale (Regressione Lineare)**: Viene fittato un modello lineare con tutte e 100 le covariate. 
4. **Il Problema (Overfitting)**: L'errore in Train è minuscolo, ma quello in Test è gigantesco. Il modello non sa generalizzare, ha imparato a memoria. Inoltre, per via dell'altissima correlazione, i coefficienti sono impossibili da interpretare.

## 5. Focus: Il Sample Splitting
La divisione dei dati è il cuore dell'approccio Machine Learning.
```r
set.seed(253143)
idx_tr <- sample(c(T,F), nrow(meatspec), replace = T, prob = c(.7,.3))
idx_tr <- which(idx_tr)
tr <- meatspec[ idx_tr, ]
te <- meatspec[-idx_tr, ]
```
* `set.seed()` fissa la generazione di numeri casuali garantendo che lo split sia sempre replicabile e identico a ogni esecuzione.
* `sample()` lancia una moneta "truccata" al 70%/30% assegnando `TRUE` (destinato al training) o `FALSE` (destinato al test) per ogni riga.
* `which()` trasforma la sequenza di `TRUE`/`FALSE` nei numeri di riga corrispondenti.
* Infine le righe selezionate popolano il Training set `tr`, mentre le righe escluse (`-idx_tr`) popolano il Test set `te`.
