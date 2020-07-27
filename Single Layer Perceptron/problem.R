# Neural Networks (EEE950)
# Pedro Vinicius A. B. de Venancio

# Clear workspace
rm(list=ls())
# Clear all plots
graphics.off()
# Clear console
cat("\014")

# Load algorithm library
source('slp.R')

# Generating data for training
#set.seed(10)
X_train <- seq(from = 0, to = 2*pi, by = 0.15)
# Adding random noise to input data
X_noise <- (runif(length(X_train))-0.5)/5
X_train <- X_train + X_noise

# Shuffling data
shuffle <- sample(length(X_train))
X_train <- X_train[shuffle]
X_train <- as.matrix(cbind(1, X_train))
# Adding random noise to output data
Y_train <- sin(X_train[,2])
Y_noise <- (runif(length(Y_train))-0.5)/5
Y_train <- Y_train + Y_noise
Y_train <- as.matrix(Y_train, nrow = 1)

# Plot training data
x_limits <- c(0, 2*pi)
y_limits <- c(-1, 1)
plot(X_train[,2], Y_train, col = 'blue', xlim = x_limits, ylim = y_limits, xlab = 'x', ylab = 'y')

# Generating data for test
X_test <- seq(from = 0, to = 2*pi, by = 0.01)
X_test <- as.matrix(cbind(1, X_test))
Y_test <- sin(X_test[,2])
Y_test <- as.matrix(Y_test)
lines(X_test[,2], Y_test, col = 'red')
legend <- c('Train', 'Test')
legend('topright', cex = 0.8, legend = legend, col = c('blue','red'), pch = c('o','-'))

# Number of neurons in the hidden layer
n_neurons <- 5

# Find weights that best fit the data
weights <- slp(X_train, Y_train, n_neurons = n_neurons, alpha = 0.01, max_epochs = 45000)
W_hidden <- matrix(unlist(weights[1]), ncol = n_neurons)
W_output <- matrix(unlist(weights[2]))

# Mapping from input layer to hidden layer
A <- sigmoid(X_test %*% W_hidden)
# Predictions
Y_hat <- A %*% W_output

# Compute test loss
test_loss <- mspe(Y_test, Y_hat)
sprintf('Test loss: %f.', test_loss)

# Plot a sine approximation
plot(X_test[,2], Y_test, type = 'l', col = 'blue', xlim = x_limits, ylim = y_limits, xlab = 'x', ylab = 'y')
lines(X_test[,2], Y_hat, lty = 2, col = 'red')
legend <- c('Sine', 'Approximation')
legend('topright', cex = 0.8, legend = legend, col = c('blue','red'), pch = c('_','-'))
