# =========================================================================
# Soluzione Esercizio Finale: Random Forest
# =========================================================================

# Carica i pacchetti necessari
library(caret)
library(randomForest)

# Assicuriamoci di aver caricato l'ambiente di lavoro dell'esercizio
# load("salvailgiusto_day04.RData")
# N.B. Si suppone che 'tr_cl', 'te_cl' e 'ctrl' (o 'tree1') siano presenti nell'environment

# 1. Impostiamo il controllo della Cross-Validazione
# Usiamo lo stesso 5-fold CV usato per l'albero per garantire un confronto leale
ctrl <- trainControl(method = "cv", number = 5)

# 2. Addestramento del modello Random Forest
# Il metodo "rf" in caret usa il pacchetto randomForest. 
# 'tuneLength = 5' significa che caret proverà 5 diversi valori per 'mtry'
# (il numero di variabili sorteggiate casualmente a ogni split).
set.seed(43113) # Impostiamo lo stesso seed per riproducibilità
mod_rf <- train(fat_class ~ ., 
                data = tr_cl, 
                method = "rf", 
                trControl = ctrl,
                tuneLength = 5,
                importance = TRUE) # Fondamentale per calcolare la varImp

# Diamo un'occhiata alle performance in validazione incrociata
print(mod_rf)
plot(mod_rf, main = "Ottimizzazione iperparametro mtry (Random Forest)")

# 3. Importanza delle variabili
# A differenza del singolo albero che si fossilizza su poche variabili chiave, 
# la RF "costringe" gli alberi a esplorare variabili diverse ad ogni split.
rf_imp <- varImp(mod_rf)
print(rf_imp)
plot(rf_imp, top = 20, main = "Random Forest: Importanza delle Variabili (Top 20)")

# 4. Previsioni sul Test Set e Matrice di Confusione
pred_rf_te <- predict(mod_rf, te_cl)

cat("\n--- Matrice di Confusione: Random Forest ---\n")
cm_rf <- confusionMatrix(pred_rf_te, te_cl$fat_class)
print(cm_rf)

# 5. Confronto diretto tra Albero Singolo e Random Forest
# (Questo blocco funziona se nello script precedente hai ancora 'tree1' in memoria)
if(exists("tree1")) {
  cat("\n--- Confronto Resamples: Albero vs Random Forest ---\n")
  models_comparison <- list(
    CART_Singolo = tree1,
    RandomForest = mod_rf
  )
  
  # Estraiamo le metriche calcolate sui fold della cross-validazione
  resamps <- resamples(models_comparison)
  
  # Riassunto numerico
  print(summary(resamps))
  
  # Grafico comparativo (Boxplot)
  # bwplot(resamps, metric = "Accuracy", main = "Confronto Accuratezza: CART vs RF")
} else {
  cat("\nAttenzione: il modello 'tree1' non è in memoria. Esegui prima il codice del CART per il confronto tramite resamples().\n")
}
