# Spiegazione del Classifiers Zoo - Data Mining e Machine Learning

Questo documento esplora lo script `DISD - Classifiers Zoo.R`. Lo scopo di questo file è prettamente visivo: mostra come si comportano diversi algoritmi di classificazione (lo "Zoo") quando si trovano ad affrontare un dataset molto particolare.

## 1. Il Dataset e la Funzione di Plot
* Viene generato un dataset bidimensionale tramite `mlbench.circle(100)`. Questo genera punti di due classi disposti a forma di **cerchi concentrici** (una classe al centro, l'altra ad anello tutto intorno). 
* Questo dataset è il nemico naturale dei modelli lineari, poiché è impossibile separare il cerchio interno da quello esterno tirando una singola linea retta.
* Viene creata una funzione personalizzata `decisionplot()` che colora lo sfondo del grafico in base alle previsioni del modello, permettendoci di vedere il "confine di decisione" (Decision Boundary) disegnato dall'algoritmo.

## 2. Lo "Zoo" dei Classificatori: Come tracciano i confini?

Il codice passa in rassegna numerosi modelli, mostrando pregi e difetti di ciascuno:

### A. k-Nearest Neighbors (kNN)
* **k = 1:** Il confine è frastagliato e crea "isole" per inglobare ogni singolo punto anomalo. È il classico esempio visivo di **Overfitting**.
* **k = 10:** Guardando i 10 vicini più prossimi, il confine diventa molto più morbido e liscio, ignorando i piccoli rumori.

### B. Modelli Lineari: Naive Bayes, LDA, Regressione Logistica
* **LDA (Linear Discriminant Analysis) e Logistic Regression:** Possono tracciare *solo linee rette*. Su questo dataset falliscono miseramente, tagliando il cerchio a metà e sbagliando moltissime previsioni.
* **QDA (Quadratic Discriminant Analysis):** A differenza della LDA, la QDA può disegnare sezioni coniche (parabole, ellissi, cerchi). Su questo dataset è perfetta.

### C. Alberi Decisionali ed Ensemble
* **CART (Singolo Albero):** Gli alberi decisionali possono fare solo tagli perpendicolari agli assi X e Y. Il loro confine di decisione sarà sempre "a gradoni" (una serie di rettangoli). 
* **CART Overfittato:** Abbassando il parametro `cp`, l'albero fa tagli minuscoli per isolare ogni singolo punto di rumore in minuscoli quadratini.
* **Random Forest e C5.0:** Essendo insiemi di alberi, i loro "gradini" sono molto più smussati e riescono ad approssimare la forma circolare in modo molto più robusto e preciso.

### D. Support Vector Machines (SVM) e i Kernel
Ricollegandosi al file del "Kernel Trick", qui si vede l'effetto pratico dei vari Kernel:
* **Linear:** Come la logistica, tira una linea dritta e fallisce.
* **Radial (RBF):** Grazie al Kernel Trick (la proiezione in dimensioni superiori), proietta i punti come se formassero una "montagna" e taglia la punta, creando un confine perfettamente circolare in 2D.
* **Polynomial / Sigmoid:** Disegnano forme geometriche complesse e curve a seconda del grado del polinomio.

### E. Reti Neurali (Shallow Neural Networks con `nnet`)
Viene testata una rete neurale "superficiale" variando il numero di neuroni (nodi) nel livello nascosto (`size`):
* **size = 1:** Si comporta quasi come un modello lineare.
* **size = 2 / size = 4:** Inizia a piegare il confine, creando fasce o poligoni per intrappolare il cerchio centrale.
* **size = 10:** Con sufficiente "potenza mentale" (10 neuroni), la rete riesce a modellare perfettamente la curvatura del cerchio interno. Dimostra visivamente come reti più grandi possano catturare pattern sempre più complessi.
