# Spiegazione Day 4 e 5 - Data Mining e Machine Learning

Questo documento raccoglie i chiarimenti relativi al file `day04-05.R`, che combina la classificazione vista nel Day 3 con il dataset spettrometrico visto nei Day 1 e 2.

## 1. Da Regressione a Classificazione (Binarizzazione)
L'obiettivo iniziale era prevedere l'esatta quantità di grasso (`fat`). Ora il problema viene semplificato in un task di classificazione: distinguere la carne "magra" ("low") da quella "grassa" ("high").
Ci sono due modi mostrati per farlo:
* **Occhiometrico**: Si guarda l'istogramma, si nota un "buco" intorno al valore 22, e si usa la funzione `cut()` per dividere manualmente in < 22 (low) e > 22 (high).
* **Algoritmico (K-Means Clustering)**: Si usa un algoritmo di apprendimento non supervisionato (il k-means con `centers = 2`) per far trovare al computer in automatico i due gruppi ottimali.
Viene quindi creato un nuovo dataset `meatclass` dove la colonna numerica `fat` è stata rimpiazzata da quella categoriale `fat_class`.

## 2. Il framework `caret` per gli Alberi
Viene utilizzato in modo massiccio il pacchetto `caret` per addestrare un albero (`method = "rpart2"`). 
Vengono addestrati due modelli:
1. `tree1`: Lascia decidere a `caret` i parametri migliori usando una Cross-Validazione a 5-fold.
2. `tree2`: Forza l'algoritmo a testare una griglia specifica di parametri (nel dettaglio, fa provare alberi con una profondità massima `maxdepth` che va da 1 a 20).

## 3. Visualizzazione ed Importanza delle Variabili
* Usando il pacchetto `rattle` e la funzione `fancyRpartPlot()`, l'albero risultante viene disegnato in modo molto più accattivante e leggibile rispetto ai plot base di R.
* Viene introdotta `varImp()` (Variable Importance). Questa funzione calcola quali sono le frequenze dello spettro che contribuiscono maggiormente a "separare" la carne grassa da quella magra, aiutandoci a interpretare il modello (un grande vantaggio degli alberi rispetto alle regressioni lineari su variabili iper-correlate!).

## 4. Valutazione Avanzata: La Matrice di Confusione
Non basta più calcolare l'errore di classificazione totale. In problemi complessi o sbilanciati, fare un errore classificando come "magro" un pezzo grasso potrebbe costare di più che classificare "grasso" un pezzo magro.
Viene introdotta `confusionMatrix()` che calcola:
* **Accuracy**: Proporzione di previsioni corrette.
* **Sensitivity (Sensibilità / Recall)**: Capacità di individuare correttamente la classe di interesse (es. i positivi veri / tutti i veri positivi).
* **Specificity (Specificità)**: Capacità di evitare falsi allarmi (es. negativi veri / tutti i veri negativi).
* **Kappa**: Una misura che aggiusta l'accuratezza in base al caso.

Alla fine, viene mostrato come modificare il `trainControl` (con `twoClassSummary`) per ottimizzare il modello usando metriche di probabilità come l'area sotto la curva ROC invece della semplice accuratezza, lasciando come esercizio finale l'applicazione di una Random Forest (RF).
