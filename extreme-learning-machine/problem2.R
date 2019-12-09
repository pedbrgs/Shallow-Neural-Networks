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

# Load algorithm library
source('elm.R')

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
# Number of neurons from hidden layer
n_neurons <- 105

# Bias
bias <- seq(1, 1, length = n_samples)

# Input matrix
X <- cbind(bias, BreastCancer[2:10])
X <- matrix(as.integer(unlist(X)), nrow = n_samples, ncol = n_features + 1)

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

# Find weights that best fit the data with training algorithm of extreme learning machines
W <- elm(X_train, Y_train, n_neurons = n_neurons)
W_1 <- matrix(data = unlist(W[1]), nrow = n_features + 1, ncol = n_neurons)
W_2 <- matrix(data = unlist(W[2]), nrow = n_neurons, ncol = 1)

# Predictions with final weights
A_1 <- tanh(X_test %*% W_1)
Y_hat_t <- A_1 %*% W_2

# Confusion matrix
print(confusion.matrix(Y_test, Y_hat_t))
# Estimates six measures of accuracy 
# (AUC, Omission Rate, Sensitivity, Specificity, Proportion Correctly Identified and Kappa)
print(accuracy(Y_test, Y_hat_t))
# Mean Squared Error (MSE)
print(mean((Y_test - Y_hat_t)^2))