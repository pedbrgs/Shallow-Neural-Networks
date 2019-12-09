# Neural Networks (EEE950)
# Pedro Vinicius A. B. de Venancio

# Clear workspace
rm(list=ls())
# Clear all plots
graphics.off()
# Clear console
cat("\014")

# Import library
library(latex2exp)

# Load data
load('data2classXOR.txt')

# Load algorithm
source('kmeans.R')
source('rbf.R')

# Number of clusters
k_clusters = 5

# Color vector for plotting
palette <- c('blue', 'red', 'black', 'pink', 'yellow', 'gray', 'orange', 'green', 'purple', 'brown', 'violet', 'gold', 'cyan', 'magenta')

# Find k-cluster centroids
cluster <- kmeans(X, k_clusters)

# Find weights that best fit the data with training algorithm of radial basis function network
W <- rbf(X, Y, cluster, k_clusters)

# Plot different colors per cluster
for(k in 1:k_clusters){
  if(k == 1){
    C <- do.call(rbind, cluster[k])
    plot(C[,1], C[,2], col = palette[k], xlim = c(0,6), ylim = c(0,6), xlab = TeX('x_1'), ylab = TeX('x_2'))
  }
  else{
    C <- do.call(rbind, cluster[k])
    points(C[,1], C[,2], col = palette[k])
  }
}

# Contour plot
contour_plot2D(X, W, cluster, axis_lim = c(0,6), step = 0.1)
