# Neural Networks (EEE950)
# Pedro Vinicius A. B. de Venancio

# Clear workspace
rm(list=ls())
# Clear all plots
graphics.off()
# Clear console
cat("\014")

# Import libraries
library(corpcor)
library(latex2exp)
library(ggplot2)

# Load data
load('data2classXOR.txt')

# Load algorithm
source('elm.R')

# Number of neurons
n_neurons <- 10
# Number of samples
n_samples <- dim(X)[1]
# Number of features
n_features <- dim(X)[2]

# Training set with bias
bias <- matrix(1, nrow = n_samples, ncol = 1)
X <- cbind(bias, X)

# Plot data
plot(X[1:n_samples/2, 2], X[1:n_samples/2, 3], col = 'blue', xlim = c(0,6), ylim = c(0,6), xlab = TeX('x_1'), ylab = TeX('x_2'))
points(X[(n_samples/2+1):n_samples, 2], X[(n_samples/2+1):n_samples, 3], col = 'red')

# Training algorithm of extreme learning machine
W <- elm(X, Y, n_neurons = n_neurons)
W_1 <- matrix(data = unlist(W[1]), nrow = n_features + 1, ncol = n_neurons)
W_2 <- matrix(data = unlist(W[2]), nrow = n_neurons, ncol = 1)
Y_hat <- sign(matrix(data = unlist(W[3]), nrow = n_samples, ncol = 1))

# Test set with bias
X_t <- cbind(bias, X_t)

# Testing weights obtained by the extreme learning machine
A_1 <- tanh(X_t %*% W_1)
Y_hat_t <- sign(A_1 %*% W_2)

# Contour plot
contour_plot2D(X, W_1, W_2, axis_lim = c(0,6), step = 0.01)
