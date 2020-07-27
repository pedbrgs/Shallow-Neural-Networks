# Neural Networks (EEE950)
# Pedro Vinicius A. B. de Venancio
# K-Means Clustering

# X: input matrix with dimension n_samples x n_features
# k_clusters: number of clusters
# cluster: list containing all clusters with their respective samples
# centroid: matrix to store centroids of all clusters (k_clusters x n_features)

kmeans <- function(X, k_clusters){
  
  # Number of samples
  n_samples <- dim(X)[1]
  # Number of features
  n_features <- dim(X)[2]
  
  # List of clusters
  cluster <- list()
  # Matrix to store centroids of all clusters (k_clusters x n_features)
  centroid <- matrix(nrow = k_clusters, ncol = n_features)
  
  # Initial clusters assignments
  Y <- sample(k_clusters, n_samples, replace = TRUE)

  # Compute initial cluster centroids
  for(k in 1:k_clusters){
    index <- which(Y == k)
    cluster[k] <- list(X[index,])
    centroid[k,] <- colMeans(do.call(rbind, cluster[k]))
  }
  
  changes <- TRUE
  # Keep iterating until there is no change to the centroids
  while(changes){
    
    Y_last <- Y
    
    # Assign sample n to the nearest cluster
    for(n in 1:n_samples){
      min_dist <- Inf
      dist <- 0
      for(k in 1:k_clusters){
        if(!all(is.nan(centroid[k,]))){
          dist <- sqrt(sum((X[n,] - centroid[k,])^2))
          if(dist < min_dist){
            min_dist <- dist
            Y[n] <- k
          }
        }
        # Fix empty clusters
        else{
          centroid[k,] <- X[n,]
          Y[n] <- k
        }
      }
    }
    
    # Update cluster centroids
    for(k in 1:k_clusters){
      index <- which(Y == k)
      cluster[k] <- list(X[index,])
      centroid[k,] <- colMeans(do.call(rbind, cluster[k]))
    }
    
    # There was no change in the clusters
    if(all(Y_last == Y)){
      changes <- FALSE
    }
    
  }
  
  return(cluster)
  
}