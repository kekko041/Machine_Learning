
## Saving data
# save.image(file = "network_project.RData")

## Loading data
# load("network_project.RData")

# Importing dataset

network <- read.delim(
  file.path("materiale", "network", "network", "netintru.txt"),
  header = TRUE,
  stringsAsFactors = TRUE
)

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

sapply(
  network[sapply(network, is.numeric)],
  sd
)

table(network$is_hot_login)
# is_hot_login si può eliminare
network$is_hot_login <- NULL


###
# Creazione dataset di traning e test
###

set.seed(123)

## Faccio uno split bilanciato anche perché ho delle classi molto sbilanciate verso normal e dos)
library(caret)

idx <- createDataPartition(
  network$result,
  p = 0.75,
  list = FALSE,
  times = 1
)

tr <- network[idx, ]
te <- network[-idx, ]

prop.table( table(network$result) )

prop.table(table(tr$result))
prop.table(table(te$result))


###
# CARET
###

library(caret)
library(rpart)

ctrl <- trainControl(
  method = "cv",
  number = 5
)

set.seed(123)

cart_model <- train(
  result ~ .,
  data = tr,
  method = "rpart",
  trControl = ctrl
)

cart_model


cart_model_2 <- train(
  result ~ .,
  data = tr,
  method = "rpart2",
  trControl = ctrl
)

cart_model_2


#Nel dataset analizzato, il miglior risultato è stato ottenuto con un valore relativamente basso di cp, 
#indicando che una struttura decisionale più dettagliata è necessaria per distinguere correttamente le 
#diverse tipologie di traffico di rete.

plot(cart_model)

library(rattle)
fancyRpartPlot(cart_model$finalModel)


plot( varImp(cart_model), top = 20 )
varImp(cart_model)


pred_cart <- predict(
  cart_model,
  te
)

cm_cart <- confusionMatrix(
  pred_cart,
  te$result
)

cm_cart

###
# Random Forest
###

library(randomForest)

set.seed(123)

rf_model <- train(
  result ~ .,
  data = tr,
  method = "rf",
  trControl = ctrl,
  #tuneLength = 10,
  verbose = TRUE
)

rf_model

plot(rf_model)

pred_rf <- predict(
  rf_model,
  te
)

cm_rf <- confusionMatrix(
  pred_rf,
  te$result
)

cm_rf

imp_rf <- varImp(rf_model)

imp_rf


####
# Matrice di correlazione
# Copia del dataset
network_num <- network

# Conversione delle categoriche in numeriche
network_num$protocol_type <- as.numeric(as.factor(network_num$protocol_type))
network_num$service       <- as.numeric(as.factor(network_num$service))
network_num$flag          <- as.numeric(as.factor(network_num$flag))
# Escludo la variabile target dalla matrice di correlazione (che ha senso solo tra predittori)
network_num$result <- NULL
dim(network_num)

cor_matrix <- cor(network_num)

dim(cor_matrix)

library(corrplot)

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.cex = 0.7,
  tl.col = "black"
)