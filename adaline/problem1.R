# Neural Networks (EEE950)
# Pedro Vinicius A. B. de Venancio

# Clear workspace
rm(list=ls())
# Clear all plots
graphics.off()
# Clear console
cat("\014")

# Load algorithm library
source('adaline.R')

# Linear samples data

set.seed(19)
# Number of samples
n_samples <- 50
X <- runif(n_samples, min = 1, max = 100)

# Bias
bias <- seq(1, 1, length = n_samples)
X <- cbind(bias, X)

# Straight line
slope <- 5
intercept <- 10
Y <- slope*X[,2] + intercept

# Adding white Gaussian noise to linear data
noise <- rnorm(n_samples, mean = 0, sd = 25)
Y <- Y + noise

# Data visualization
plot(X[,2], Y, xlab = 'x', ylab = 'y', col = 'red')

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

# Find weights that fit the data with the adaline algorithm
W <- adaline(X_train, Y_train, eta = 0.001, max_epochs = 25000, tol = 0.03)
colnames(W) <- c('x0', 'x1')

# Predictions with final weights
Y_hat <- activation(X_test, W)

# Plot straight line
straight_line(X, W)

# Mean Squared Error (MSE)
MSE <- mean((Y_test - Y_hat)^2)
print(MSE)