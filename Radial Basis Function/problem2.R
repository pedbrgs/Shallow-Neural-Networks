# Neural Networks (EEE950)
# Pedro Vinicius A. B. de Venancio

# Clear workspace
rm(list=ls())
# Clear all plots
graphics.off()
# Clear console
cat("\014")

# Import libraries
library(mlbench)
library(SDMTools)

# Load algorithm
source('kmeans.R')
source('rbf.R')

# Load data
data('BreastCancer')

# Converting categorical labels to numbers (benign = 0, malignant = 1)
BreastCancer['Class'] <- as.integer((unlist(BreastCancer['Class'])))
BreastCancer$Class[BreastCancer$Class == 1] <- 0
BreastCancer$Class[BreastCancer$Class == 2] <- 1
BreastCancer <- na.omit(BreastCancer)

# Number of sampels after remove rows with all or some NaNs
n_samples <- dim(BreastCancer)[1]
# Subtract two because one being a character variable and 1 target class
n_features <- dim(BreastCancer)[2] - 2
# Number of clusters/neurons from hidden layer
k_clusters <- 10

# Bias
bias <- seq(1, 1, length = n_samples)

# Input matrix
X <- cbind(BreastCancer[2:10])
X <- matrix(as.integer(unlist(X)), nrow = n_samples, ncol = n_features)

# Output vector
Y <- cbind(BreastCancer['Class'])
Y <-as.vector(t(matrix(unlist(Y), ncol = 1, nrow = n_samples)))

# Splitting data for training and test
proportion <- 0.7
set.seed(10)
perm <- sample(dim(X)[1])
train_index <- perm[1:round(proportion*n_samples)]
test_index <- perm[round(proportion*n_samples + 1):n_samples]
X_train <- X[train_index,]
Y_train <- Y[train_index]
X_test <- X[test_index,]
Y_test <- Y[test_index]

# Matrix to store centroids of all clusters (k_clusters x n_features)
centroid <- matrix(nrow = k_clusters, ncol = n_features)

# Find k-cluster centroids
cluster <- kmeans(X_train, k_clusters)

# Find weights that best fit the data with training algorithm of radial basis function network
W <- rbf(X_train, Y_train, cluster, k_clusters)

# Compute cluster centroids
for(k in 1:k_clusters){
  centroid[k,] <- colMeans(do.call(rbind, cluster[k]))
}

# Estimate width of Gaussian kernels
sigma <- width_estimate(centroid)

# Mapping from input layer to hidden layer (n_samples x k_clusters + 1)
A_1 <- matrix(nrow = length(Y_test), ncol = k_clusters)
for(i in 1:length(Y_test)){
  for(j in 1:k_clusters){
    v <- norm(as.matrix((X_test[i,] - centroid[j,])))
    A_1[i,j] <- gaussian(v, sigma)
  }
}
A_1 <- cbind(1, A_1)

# Predictions with final weights
Y_hat_t <- A_1 %*% W

# Confusion matrix
print(confusion.matrix(Y_test, Y_hat_t))
# Estimates six measures of accuracy 
# (AUC, Omission Rate, Sensitivity, Specificity, Proportion Correctly Identified and Kappa)
print(accuracy(Y_test, Y_hat_t))
# Mean Squared Error (MSE)
print(mean((Y_test - Y_hat_t)^2))