# Neural Networks (EEE950)
# Pedro Vinicius A. B. de Venancio
# Adaptive Linear Element (Adaline)

# X: input matrix with dimension (n_samples * proportion) x (n_features + 1)
# W: weights vector with dimension 1 x (n_features + 1)
# Y: labels vector with dimension 1 x (n_samples * proportion)
# Y_hat: predictions vector with dimension 1 x (n_samples * proportion)
# proportion: percentage used to split data into training and test sets

# Activation function
activation <- function(X, W){
  z = t(X %*% t(W))
  return(z)
}

# Adaline algorithm
adaline <-function(X, Y, eta = 0.001, max_epochs = 100, tol = 0.001){
  
  # Number of features
  n_features <- dim(X)[2]
  # Random initial weights
  W <- t(rnorm(n_features))

  # Convergence parameters
  MSE <- Inf
  epoch <- 0

  # Until the number of epochs is reached and the error is less than tolerance
  while(epoch < max_epochs && MSE > tol){

    # Predictions with current weights
    Y_hat <- activation(X, W)
    
    # Mean Squared Error (MSE)
    MSE <- mean((Y - Y_hat)^2)
    
    # Convergence statistics
    cat(sprintf('Epoch: %.0f. Mean Squared Error (MSE): %.6f.\n', epoch, MSE))
    
    # Normalized error (alpha-LMS rule)
    error <- (Y - Y_hat)/sum((Y - Y_hat)^2)
    # Weights updates
    W <- W + eta*(error %*% X)
    
    # Update number of epochs
    epoch <- epoch + 1
    
  }
  return(W)
}

# Plot straight line
straight_line <- function(X, W){
  classifier <- W[2]*sort(X[,2]) + W[1]
  lines(sort(X[,2]), classifier)
}