if (!require("cluster")) install.packages("cluster")
library(cluster)

library(readr)
dataset <- read_csv("C:/Users/emont/OneDrive/Escritorio/Tesis/MILP/Datasets/Datasets/Medium/Migration.csv")

X <- dataset[, -c(3,10)]
y <- dataset$GIM_2000

# Define the number of elements in each cluster
k = length(table(y))

# Calculate distance matrix
if (!require("proxy")) install.packages("proxy")
library(proxy)
D <- proxy::dist(as.matrix(X), method = "cosine")
D = as.matrix(D)
distance = D
D1 = as.dist(distance)

#===========================================================================================================
# AHC Algorithm
r <- nrow(D) 
hc <- hclust(D1, method = "complete")
label_pred <- cutree(hc, k)

#================================================================================================================
X_mat <- as.matrix(X)
X_norms <- sqrt(rowSums(X_mat^2)) + 1e-10
X_norm <- X_mat / X_norms

u_labels <- sort(unique(label_pred))
centroids <- matrix(0, nrow = length(u_labels), ncol = ncol(X_mat))
for(i in seq_along(u_labels)) {
  mask <- (label_pred == u_labels[i])
  if(sum(mask) == 1) centroids[i,] <- X_mat[mask,] 
  else centroids[i,] <- colMeans(X_mat[mask,])
}

# Calculate Cosine Distances
c_norms <- sqrt(rowSums(centroids^2)) + 1e-10
c_norm <- centroids / c_norms
dists <- 1 - tcrossprod(X_norm, c_norm)
dists[dists < 0] <- 0

# Calculate a(i) and b(i)
idx <- cbind(1:nrow(X), match(label_pred, u_labels))
a_i <- dists[idx]      
dists[idx] <- Inf       
b_i <- apply(dists, 1, min) 
sil_compatible <- mean((b_i - a_i) / pmax(a_i, b_i), na.rm = TRUE)

# 4. Final Metrics
cat("--------------------------------------------------\n")
cat("Silhouette (Centroid-based Logic):", sil_compatible, "\n")
cat("ARI                            :", ARI(y, label_pred), "\n")
cat("AMI                            :", AMI(y, label_pred), "\n")
cat("NMI                            :", NMI(y, label_pred), "\n")
cat("--------------------------------------------------\n")

# Counts
cat("\nGround Truth Groups:\n"); print(table(y))
cat("\nAHC Groups:\n"); print(table(label_pred))
