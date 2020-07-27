# Neural Networks (EEE950)
# Pedro Vinicius A. B. de Venancio
# Single Layer Perceptron

# X: input matrix with dimension (n_samples * proportion) x (n_features + 1)
# W_hidden: Hidden layer weights with dimension (n_features+1) x n_neurons
# W_output: Output layer weights with dimension n_neurons x n_outputs
# Y: labels vector with dimension (n_samples * proportion) x n_outputs
# Y_hat: predictions vector with dimension (n_samples * proportion) x n_outputs
# n_neurons: number of neurons from hidden layer

# Sigmoid function
sigmoid <-function(x){
  return(1.0/(1.0+exp(-x)))
}

# Sigmoid derivative
sigmoid_derivative <- function(x){
  return(x*(1.0-x))
}

# Mean Squared Percentage Error (MSPE)
mspe <- function(Y, Y_hat){
  norm_mse <- ((Y-Y_hat)/Y)^2
  norm_mse[!is.finite(norm_mse)] <- NA 
  return(colMeans(norm_mse, na.rm = TRUE))
}

# L2 loss function - Mean Squared Error (MSE)
loss_function <- function(Y, Y_hat){
  return(sum((Y-Y_hat))^2)
}

# L2 derivative (Mean Squared Error derivative)
loss_derivative <- function(Y, Y_hat){
  return(2*(Y-Y_hat))
}

# One-Hidden-Layer training algorithm
slp <- function(X, Y, n_neurons, max_epochs = 50, alpha = 0.01){
  
  # Number of samples
  n_samples <- dim(X)[1]
  # Number of features
  n_features <- dim(X)[2]
  # Number of outputs
  n_outputs <- dim(Y)[2]
  # Number of epochs
  n_epochs <- 0
  
  # Random weights of input layer (n_features x n_neurons)
  W_hidden <- matrix(runif(n_features * n_neurons, min = -0.5, max = 0.5), nrow = n_features, ncol = n_neurons, byrow = TRUE)

  # Random weights of hidden layer (n_neurons x n_outputs)
  W_output <- matrix(runif(n_neurons, min = -0.5, max = 0.5), ncol = n_outputs)

  # Hidden layer mapping matrix
  A <- matrix(nrow = n_samples, ncol = n_neurons)
  # Prediction matrix
  Y_hat <- matrix(nrow = n_samples, ncol = n_outputs)
  # Vector for storing errors during training
  train_loss <- matrix(nrow = 1, ncol = n_epochs)

  # Until the maximum number of epochs is reached
  while(n_epochs < max_epochs){
  
    # Forward pass
    A <- sigmoid(X %*% W_hidden)
    Y_hat <- A %*% W_output
    
    # Backward pass
    dW_output <- t(A) %*% (loss_derivative(Y, Y_hat) * 1)
    dW_hidden <- t(X) %*% (((loss_derivative(Y, Y_hat) * 1) %*% t(W_output))*sigmoid_derivative(A))
    
    # Update weights
    W_output <- W_output + alpha*dW_output
    W_hidden <- W_hidden + alpha*dW_hidden

    # Increase number of epochs
    n_epochs <- n_epochs + 1
    
    # Compute training loss
    train_loss[n_epochs] <- loss_function(Y, Y_hat)
    print(sprintf('Training loss: %f.', train_loss[n_epochs]))
  }
  
  # Plot training loss
  plot(train_loss, type = 'l', xlab = 'Epoch', ylab = 'Loss', col = 'red')
  # Print average training loss
  print(sprintf('Avg Training loss: %f.', mean(train_loss)))
  
  return(list(W_hidden, W_output))
  
}
