# Soluzione Esercizio Finale: Random Forest

Questo documento illustra i concetti e i risultati dell'esercizio finale proposto nel file `day04-05.R`, che richiede l'addestramento e il confronto di una **Random Forest (RF)** con un singolo albero CART.

## 1. Che cos'è una Random Forest?
Il problema principale di un singolo albero decisionale (CART) è la sua **alta varianza**: è molto instabile e sensibile a piccole modifiche nel Training set. 
La Random Forest risolve questo problema essendo un *Ensemble Model* (un "coro" di modelli):
* Genera centinaia di alberi (spesso 500 o 1000).
* **Bagging**: Ogni albero viene addestrato su un campione leggermente diverso (estratto con reinserimento).
* **Random Subspace**: Ad ogni divisione (split) di un ramo, l'albero non può scegliere tra tutte le 100 variabili dello spettro, ma solo su un piccolo sottoinsieme casuale (il parametro `mtry`).
La previsione finale si ottiene tramite voto a maggioranza.

## 2. Il parametro di Tuning: `mtry`
Nel codice, abbiamo usato `tuneLength = 5` in `caret::train(..., method="rf")`. Questo fa sì che la Cross-Validazione cerchi il miglior valore per `mtry` (quante variabili guardare ad ogni split). Solitamente per la classificazione si parte dalla radice quadrata del numero totale di predittori (es. $\sqrt{100} = 10$).

## 3. L'importanza delle Variabili (`varImp`)
Questa è la parte concettualmente più interessante dell'esercizio rispetto a quanto visto con la regressione lineare (Day 1) e il singolo albero:
* **Regressione Lineare:** Abbiamo visto che con 100 variabili fortemente correlate, i coefficienti esplodono e non si capisce nulla.
* **Singolo Albero:** Poiché tutte le variabili contengono informazioni simili (il profilo dello spettro è una curva morbida), l'albero sceglierà una variabile a caso tra quelle simili per fare il primo split, scartando del tutto le altre "gemelle". L'importanza si concentrerà su 1 o 2 variabili, ignorando le altre (fenomeno del *masking*).
* **Random Forest:** Poiché l'algoritmo "nasconde" alcune variabili ad ogni split, prima o poi costringerà gli alberi a usare tutte le parti dello spettro. Di conseguenza, il grafico della `varImp()` mostrerà un'importanza molto più **"distribuita"** su diverse frequenze rispetto al singolo albero. Troverai un gruppetto allargato di feature che contribuiscono in modo significativo al modello.

## 4. Confronto delle Performance (`confusionMatrix` e `resamples`)
Eseguendo l'esercizio, noterai tipicamente due cose:
1. **L'accuratezza (e il Kappa) della Random Forest è superiore** rispetto al singolo albero, grazie all'effetto stabilizzante del voto a maggioranza dell'ensemble (drastica riduzione della varianza).
2. Tramite la funzione `resamples()`, il boxplot (e le deviazioni standard delle fold) mostrerà che la Random Forest non solo ha un'accuratezza media più alta, ma anche una **"scatola" più stretta**, a dimostrazione della sua formidabile stabilità sui diversi tagli di training set.
