# Spiegazione Day 2 - Data Mining e Machine Learning

Questo documento raccoglie i chiarimenti relativi al file `day02.html` (e al relativo codice R).

## 1. Il problema di partenza
Si riprende dal cliffhanger del Day 1: inserire 100 variabili spettrometriche altissimamente correlate in un modello lineare semplice porta all'**Overfitting** (ottime prestazioni sul Training, pessime sul Test) e alla **Multicollinearità** (impossibilità di interpretare i parametri). L'obiettivo del Day 2 è trovare modelli capaci di superare queste sfide, generalizzando in maniera più robusta.

## 2. Tentativo 1: Eliminare le variabili "doppione"
Visto che tante variabili dicono "la stessa cosa", una strategia è semplicemente rimuovere quelle troppo correlate.
* Viene introdotto il framework `caret`.
* Si utilizza la funzione `findCorrelation()` per scartare una parte di predittori in base a un **cut-off** (es. 0.99).
* La vera sfida è capire quale cut-off utilizzare senza "sbirciare" o compromettere i dati di Test.

## 3. Il Trade-off Bias-Varianza e la K-Fold Cross-Validation (CV)
* Il professore scrive "a mano" una validazione incrociata a 3-fold (3-CV).
* I dati di addestramento vengono suddivisi in 3 sotto-blocchi per valutare il cut-off in modo iterativo.
* Questa fase illustra il cuore del **Bias-Variance Trade-off**:
  * Un modello con troppe poche variabili diventa troppo semplice (alta distorsione, Underfitting).
  * Un modello con troppe variabili tende a imparare a memoria (alta varianza, Overfitting).
  * La Cross-Validazione ha lo scopo di aiutarci a trovare il cut-off migliore (il giusto punto di equilibrio).

## 4. Tentativo 2: Principal Component Regression (PCR)
Eliminare variabili può scartare informazione preziosa. Un approccio diverso consiste nel **trasformare** i dati.
* La **PCA (Analisi in Componenti Principali)** prende le 100 variabili correlate e crea un numero inferiore di nuove variabili chiamate *componenti principali*, le quali sono **matematicamente scorrelate** l'una dall'altra.
* Tramite la funzione `train()` di `caret`, si fitta un modello "pcr", delegando la scelta del numero ottimo di componenti principali alla cross-validazione automatica del pacchetto.
* Questo approccio migliora nettamente l'errore sul set di Test rispetto al modello originario lineare.

## 5. Tentativo 3: Partial Least Squares (PLS)
La PCA del tentativo precedente spreme le variabili X ignorando del tutto la variabile risposta (il grasso che dobbiamo prevedere).
* Il modello **PLS (Partial Least Squares)** risolve questa debolezza: trova nuove componenti che non solo sintetizzano la matrice delle X, ma che al contempo sono le **massimamente correlate con il target Y**.
* La PLS, usando `method = "pls"` in `caret`, solitamente permette di ottenere risultati ottimi o superiori alla PCR, spesso richiedendo un numero ancora minore di componenti (e dunque un modello ancora più leggero).

## 6. Confronto Finale
Al termine della lezione, i tre modelli (LM con variabili filtrate, PCR e PLS) vengono messi a paragone usando la funzione `resamples()`. Tramite boxplot è possibile apprezzare visivamente quale metodo offra l'errore (RMSE) più basso e quale risulti essere il più stabile sulle iterazioni di validazione incrociata.
