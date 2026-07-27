##Confusion matrix per i tre modelli (per dare valore alle casistiche u2r, 12 casi su quasi 4000)

library(ggplot2)

plot_confmat <- function(cm, titolo) {
  df <- as.data.frame(cm$table)
  df$Percent <- ave(df$Freq, df$Reference, FUN = function(x) x / sum(x) * 100)
  
  ggplot(df, aes(x = Prediction, y = Reference, fill = Percent)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 3.5) +
    scale_fill_gradient(low = "lightblue", high = "darkblue", name = "% entro\nclasse reale") +
    labs(title = titolo, x = "Predetto", y = "Reale") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_confmat(cm_cart, "CART")
plot_confmat(cm_rf, "Random Forest")
plot_confmat(cm_nnet, "Rete Neurale")


##Sensitivity per classe, per modello

library(dplyr)

estrai_sens <- function(cm, nome_modello) {
  data.frame(
    Classe = gsub("Class: ", "", rownames(cm$byClass)),
    Sensitivity = cm$byClass[, "Sensitivity"],
    Modello = nome_modello
  )
}

sens_df <- bind_rows(
  estrai_sens(cm_cart, "CART"),
  estrai_sens(cm_rf, "Random Forest"),
  estrai_sens(cm_nnet, "NN")
)

ggplot(sens_df, aes(x = Classe, y = Sensitivity, fill = Modello)) +
  geom_col(position = "dodge") +
  labs(title = "Sensitivity (recall) per classe e per modello",
       subtitle = "Dove ogni modello riconosce meno le classi rare",
       y = "Sensitivity", x = NULL) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")


##Metriche globali a confronto (Accuracy + Kappa)

library(tidyr)

metriche_df <- data.frame(
  Modello = c("CART", "Random Forest", "NN"),
  Accuracy = c(cm_cart$overall["Accuracy"], cm_rf$overall["Accuracy"], cm_nnet$overall["Accuracy"]),
  Kappa    = c(cm_cart$overall["Kappa"], cm_rf$overall["Kappa"], cm_nnet$overall["Kappa"])
) %>% pivot_longer(-Modello, names_to = "Metrica", values_to = "Valore")

ggplot(metriche_df, aes(x = Modello, y = Valore, fill = Metrica)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = round(Valore, 3)), position = position_dodge(width = 0.9), vjust = -0.3) +
  labs(title = "Confronto prestazioni globali", y = NULL) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")