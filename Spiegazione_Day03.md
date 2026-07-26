# Spiegazione Day 3 - Data Mining e Machine Learning

Questo documento raccoglie i chiarimenti relativi al file `CART-screen.R` della terza lezione, che introduce per la prima volta i modelli ad albero per la **classificazione**.

## 1. Cambio Dataset: Glaucoma
Abbandoniamo temporaneamente la carne per passare a un dataset medico (`GlaucomaMVF`). L'obiettivo cambia radicalmente: non dobbiamo più prevedere un numero continuo (regressione), ma dobbiamo prevedere una **classe** o categoria (Glaucoma o Normale).

## 2. Train & Test Split (Versione Avanzata)
Viene introdotta la funzione `createDataPartition()` del pacchetto `caret`.
* A differenza della funzione base `sample()` usata nel Day 1, `createDataPartition` esegue uno **split stratificato**. Questo significa che se nel dataset originale hai il 50% di malati e il 50% di sani, la funzione si assicurerà che **anche nel Training e nel Test set** vengano rispettate esattamente queste percentuali, evitando sbilanciamenti sfortunati creati dal caso.

## 3. Alberi Decisionali (CART) con `rpart`
Viene addestrato il primo Albero di Classificazione e Regressione (CART) usando il pacchetto `rpart`.
* **Costruzione**: Il modello partiziona lo spazio in base a delle regole (es. "se la variabile X è maggiore di K, allora vai a destra, altrimenti a sinistra").
* **Il problema degli alberi (Overfitting)**: Se lasciati crescere senza limiti, gli alberi decisionali creano regole per ogni singola riga del Training set. Imparano a memoria i dati.
* **La soluzione (Pruning / Potatura)**: 
  1. Si guarda la tabella di complessità (`glaucoma.rpart$cptable`).
  2. Si cerca il parametro di complessità (`CP`) che minimizza l'errore di cross-validazione (`xerror`).
  3. Si usa la funzione `prune()` per "tagliare i rami" inutili dell'albero, rendendolo più semplice e capace di generalizzare meglio sui nuovi dati.

## 4. Alternative a rpart
Nella seconda parte dello script, il professore mostra che esistono molti algoritmi e pacchetti alternativi per creare alberi:
* **J48 (C4.5)** tramite il pacchetto `RWeka` (derivato dal celebre software Weka).
* **C5.0** (evoluzione del C4.5) tramite il pacchetto `C50`. È uno standard dell'industria.
* **ctree** (Conditional Inference Trees) tramite `party`. Usa test statistici per decidere quando splittare, risultando intrinsecamente più robusto all'overfitting senza bisogno di potature "aggressive".

## 5. Valutazione (Errore di Classificazione)
Poiché siamo in classificazione, non usiamo più l'RMSE (Root Mean Square Error). Si valuta il modello calcolando semplicemente la proporzione di risposte sbagliate sul totale: `mean(previsioni != risposte_vere) * 100`.
Come sempre, l'errore calcolato sul Training set è ottimistico, ed è fondamentale valutarlo sul Test set.
