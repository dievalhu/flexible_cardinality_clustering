# Cargar bibliotecas
if (!require("cluster")) install.packages("cluster")
if (!require("mice")) install.packages("mice")
library(cluster)
library(mice)

library(readr)
dataset <- read_csv("C:/Users/emont/OneDrive/Escritorio/Tesis/MILP/Datasets/Datasets/Medium/Water.csv")

# Seleccionar las columnas numéricas para el análisis
X <- dataset[, 1:9]
imp <- mice(X, method = "pmm", m = 5, maxit = 50, seed = 500)
X <- complete(imp, 1)
y <- dataset$Potability

# Definir el número de elementos en cada clúster
k = length(table(y))

# Calcular la matriz de distancias
if (!require("proxy")) install.packages("proxy")
library(proxy)
D <- proxy::dist(as.matrix(X), method = "cosine")
D = as.matrix(D)
distancia = D

#===========================================================================================================
#PROCESO ALGORITMO KMEDOIDS

r <- nrow(D) # Número total de documentos

cl = pam(X, k)
label_pred = cl$cluster

#================================================================================================================
X_mat <- as.matrix(X)
X_norms <- sqrt(rowSums(X_mat^2)) + 1e-10
X_norm <- X_mat / X_norms

# Calcular Centroides generados por AHC
u_labels <- sort(unique(label_pred))
centroids <- matrix(0, nrow = length(u_labels), ncol = ncol(X_mat))
for(i in seq_along(u_labels)) {
  mask <- (label_pred == u_labels[i])
  if(sum(mask) == 1) centroids[i,] <- X_mat[mask,] 
  else centroids[i,] <- colMeans(X_mat[mask,])
}

# Calcular Distancias Coseno (Punto vs Centroide)
c_norms <- sqrt(rowSums(centroids^2)) + 1e-10
c_norm <- centroids / c_norms
dists <- 1 - tcrossprod(X_norm, c_norm)
dists[dists < 0] <- 0

# Calcular a(i) y b(i)
idx <- cbind(1:nrow(X), match(label_pred, u_labels))
a_i <- dists[idx]         # Distancia a mi centroide
dists[idx] <- Inf         # Ignorar mi centroide para buscar vecino
b_i <- apply(dists, 1, min) # Distancia al centroide vecino más cercano

sil_compatible <- mean((b_i - a_i) / pmax(a_i, b_i), na.rm = TRUE)

# 4. Métricas Finales
cat("--------------------------------------------------\n")
cat("Silueta (Lógica Centroide) :", sil_compatible, "\n")
cat("ARI                        :", ARI(y, label_pred), "\n")
cat("AMI                        :", AMI(y, label_pred), "\n")
cat("NMI                        :", NMI(y, label_pred), "\n")
cat("--------------------------------------------------\n")

# Conteos
cat("\nGrupos Reales:\n"); print(table(y))
cat("\nGrupos AHC:\n"); print(table(label_pred))