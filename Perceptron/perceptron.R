# Neural Networks (EEE950)
# Pedro Vinicius A. B. de Venancio
# Perceptron

# X: input matrix with dimension (n_samples * proportion) x (n_features + 1)
# W: weights vector with dimension 1 x (n_features + 1)
# Y: labels vector with dimension 1 x (n_samples * proportion)
# Y_hat: predictions vector with dimension 1 x (n_samples * proportion)
# proportion: percentage used to split data into training and test sets

# Activation function
activation <- function(X, W, limiar = 0.5){
  z = t(X %*% t(W))
  # Heaviside function
  Y_hat = sapply(z > limiar, as.numeric)
  return(Y_hat)
}

# Training algorithm of perceptron
perceptron <- function(X, Y, eta = 0.01, max_epochs = 100){
  
  # Number of features
  n_features <- dim(X)[2]

  # Random initial weights
  W <- t(rnorm(n_features))
  # Predictions with initial weights  
  Y_hat <- activation(X, W)
  
  # Number of epochs
  epoch <- 1

  # Until the sum of the difference between the prediction and the target is zero
  while(sum(Y_hat - Y) != 0 || epoch < max_epochs){

    # Convergence statistics
    cat(sprintf('Epoch: %.0f.\n', epoch))
    
    # Weights updates
    W <- W + eta*((Y - Y_hat) %*% X)
    # Predictions with current weights
    Y_hat <- activation(X, W)
    # Update number of epochs
    epoch <- epoch + 1
    
  }
  return(W)
}

# Plot decision boundary
decision_boundary <- function(X, W){
  # Linear coefficient
  intercept <- -W[1]/W[3]
  # Angular coefficient
  slope <- -W[2]/W[3]
  # Equation of a line
  classifier <- slope*sort(X[,2]) + intercept
  lines(sort(X[,2]), classifier)
}