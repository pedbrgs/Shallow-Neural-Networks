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
source('perceptron.R')

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

# Find weights that best fit the data with the perceptron algorithm
W <- perceptron(X_train, Y_train, eta = 0.01, max_epochs = 100)

# Predictions with final weights
Y_hat <- activation(X_test, W)

# Confusion matrix
print(confusion.matrix(Y_test, Y_hat))
# Estimates six measures of accuracy 
# (AUC, Omission Rate, Sensitivity, Specificity, Proportion Correctly Identified and Kappa)
print(accuracy(Y_test, Y_hat))
# Mean Squared Error (MSE)
print(mean((Y_test - Y_hat)^2))