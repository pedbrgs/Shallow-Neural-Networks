# Neural Networks (EEE950)
# Pedro Vinicius A. B. de Venancio

# Clear workspace
rm(list=ls())
# Clear all plots
graphics.off()
# Clear console
cat("\014")

# Importy library
library(SDMTools)

# Load algorithm library
source('adaline.R')

# Load data
data = read.csv('BUILDING1paraR.DT', sep = ' ')
data <- na.omit(data)

# Data pre-processing
n_samples = dim(data)[1]
bias <- seq(1, 1, length = n_samples)

# Input matrix
X <- as.matrix(cbind(bias, data[1:14]))

# Scan from user
print('Function to be approximated::')
print('1 - Energy') 
print('2 - Hot Water')
print('3 - Cold Water') 
input <- readline()

if(strtoi(input) == 1){
  Y <- as.matrix(data['Energy'])
} else if(strtoi(input) == 2){
  Y <- as.matrix(data['Hot_Water'])
} else if(strtoi(input) == 3){
  Y <- as.matrix(data['Cold_Water'])
} else{
  print('Program terminated.')
}

if(strtoi(input) == 1 || strtoi(input) == 2 || strtoi(input) == 3){

  # Plot output
  plot(Y, type = 'l', ylim = c(0.1, 0.8))
  
  # Find weights that fit the data with the adaline algorithm
  W <- adaline(X, t(Y), eta = 0.001, max_epochs = 14000, tol = 0.001)
  
  # Function predictions
  Y_hat <- as.vector(activation(X, W))
    
  # Plot predictions
  plot(as.matrix(Y_hat), col = 'blue', type = 'l', ylab = 'Y', ylim = c(0.1, 0.8))
  
  # Plot curves comparison
  plot(as.matrix(Y_hat), col = 'blue', type = 'l', ylab = 'Y', ylim = c(0.1, 0.8))
  lines(Y, col = 'black')
  legend('topright', c('Approximation', 'Data'), col = c('blue', 'black'), lty = 1:1, cex = 0.7)
  
  # Plot curves comparison
  plot(Y, col = 'black', type = 'l', ylab = 'Y', ylim = c(0.1, 0.8))
  lines(as.matrix(Y_hat), col = 'blue')
  legend('topright', c('Data', 'Approximation'), col = c('black', 'blue'), lty = 1:1, cex = 0.7)
  
  # Mean Squared Error (MSE)
  print(mean((Y - Y_hat)^2))

}