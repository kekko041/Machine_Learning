#ESERCIZIO#

getwd() #stampo la cartella in cui mi trovo
cartella <- file.path("materiale", "network", "network")

network <- read.delim(file.path(cartella, "netintru.txt"), header = TRUE,
  stringsAsFactors = TRUE
)

#Analizzo le informazioni del dataset
dim(network)
names(network)
str(network)
head(network)
summary(network)

barplot(
  table(network$result),
  main = "Distribuzione delle classi",
  las = 2
)

prop.table(table(network$result)) # % di ogni caso 

table(network$is_hot_login) #valore sempre uguale, eliminabile per il calcolo
network$is_hot_login <- NULL

sapply(
  network[sapply(network, is.numeric)],
  sd
)

barplot(
  prop.table(table(network$result)) * 100,
  main = "Distribuzione delle classi (%)",
  las = 2,
  ylab = "Percentuale"
)

# ---- INIZIAMO! ----

set.seed(123) #inizializiamo

## Faccio uno split bilanciato anche perché ho delle classi molto sbilanciate verso normal e dos)
library(caret)


# Indici in training = TRUE
#idx <- sample(c(T,F), nrow(network), replace = T,
#                prob = c(.75,.25))

idx <- createDataPartition(network$result,p = 0.75,
  list = FALSE, times = 1)


mean(idx)#verifico s le proporzioni sono rispetttate

tr <- network[idx, ]
te <- network[-idx, ]

#verifico se la proporzione è rispettata sia nel training che testing dataset
prop.table( table(network$result) )
prop.table(table(tr$result))
prop.table(table(te$result))

# 1. Carica la libreria
library(caret)
library(nnet)
library(ggplot2)

nnet_model <- train(
  result ~ ., 
  data = tr, 
  method = "nnet", 
  tuneGrid = expand.grid(size = 4, decay = 0.1),
  trace = FALSE # Nasconde il log di addestramento
)


imp_nnet <- varImp(nnet_model)
imp_nnet

ytr_nnet <- predict(nnet_model, tr) # train
yte_nnet <- predict(nnet_model, te)

confronta_risultati <- function(pred, real) {
  # Verifica che i vettori abbiano la stessa lunghezza
  if (length(pred) != length(real)) {
    stop()
  }
  
  # Confronto elemento per elemento
  risultato <- pred == real
  return(risultato)
}

confronta_risultati(ytr_nnet, tr$result)
prop.table(table(confronta_risultati(ytr_nnet, tr$result)))
prop.table(table(confronta_risultati(yte_nnet, te$result)))

cm_nnet <-confusionMatrix(
  yte_nnet,
  te$result
)

conf_matrix <- confusionMatrix(yte_nnet, te$result)
conf_df <- as.data.frame(conf_matrix$table)
accuracy <- sum(yte_nnet == te$result) / length(te$result)

ggplot(conf_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Matrice di Confusione",
       subtitle = paste("Accuratezza:", round(accuracy * 100, 2), "%"),
       x = "Predetto", y = "Reale") +
  theme_minimal()

save(nnet_model, tr, file = "prova1nnet_esercizio.RData")
