# see Algorithm 1 in "Supervised principal component analysis: Visualization, classification andregression on subspaces and submanifolds"
# https://uwaterloo.ca/data-analytics/sites/default/files/uploads/documents/barshan_supervised_preprint.pdf

supervised_pca <- function(x, y, 
                           k = 2,
                           orthogonalize = T,
                           scale_x = T){
  stopifnot(is.matrix(x), is.matrix(y),
            nrow(x) == nrow(y))
  
  if(scale_x) x <- scale(x)
  n <- nrow(x)
  H <- diag(n) - matrix(1/n, nrow = n, ncol = n)
  half_mat <- crossprod(y, H %*% x)
  Q <- crossprod(half_mat)
  eigen_res <- eigen(Q)
  U <- Re(eigen_res$vectors[,1:k])
  
  res <- x %*% U
  
  if(orthogonalize){
    svd_res <- svd(res)
    rotation_mat <- svd_res$v
    U <- U %*% rotation_mat
    res <- x %*% U
  }
  
  colnames(res) <- paste0("SPCA_", 1:k)
  colnames(U) <- paste0("SPCA_", 1:k)
  rownames(U) <- colnames(x)
  
  list(dimred = res, U = U)
}

supervised_pca_fast <- function(x, y, k = 2,
                                orthogonalize = TRUE,
                                scale_x = TRUE,
                                use_small_c = TRUE) {
  stopifnot(is.matrix(x), is.matrix(y), nrow(x) == nrow(y))
  
  if (scale_x) x <- scale(x)  # centers & scales columns
  
  # Compute Y^T H X without H
  mu_x <- colMeans(x)              # p
  n_c  <- colSums(y)               # c (class counts)
  half_mat <- crossprod(y, x) - tcrossprod(n_c, mu_x)  # c x p
  
  # Get top k right singular vectors of half_mat (p x k loadings)
  if (use_small_c && ncol(y) <= ncol(x)) {
    # Solve tiny c x c eigenproblem, then back out V (right singular vecs)
    B <- half_mat %*% t(half_mat)                # c x c
    ee <- eigen(B, symmetric = TRUE)
    s  <- sqrt(pmax(ee$values[1:k], 0))          # singular values
    U_L <- ee$vectors[, 1:k, drop = FALSE]       # left sing vecs (c x k)
    # right sing vecs: V = t(half_mat) %*% U_L %*% diag(1/s)
    U  <- t(half_mat) %*% sweep(U_L, 2, ifelse(s > 0, s, 1), "/")
  } else {
    # Direct truncated SVD (base R computes only needed right vecs)
    sv <- svd(half_mat, nu = 0, nv = k)
    U  <- sv$v                                   # p x k
  }
  
  # Scores
  res <- x %*% U                                  # n x k
  
  # Optional: rotate scores to be orthogonal in sample space
  if (orthogonalize) {
    svd_res <- svd(res)
    rot <- svd_res$v
    U   <- U %*% rot
    res <- x %*% U
  }
  
  colnames(res) <- paste0("SPCA_", seq_len(k))
  colnames(U)   <- paste0("SPCA_", seq_len(k))
  rownames(U)   <- colnames(x)
  
  list(dimred = res, U = U)
}


form_onehot_classification_mat <- function(y){
  uniq_val <- sort(unique(y))
  k <- length(uniq_val)
  n <- length(y)
  
  mat <- matrix(0, nrow = n, ncol = k)
  for(j in 1:k){
    mat[which(y == uniq_val[j]),j] <- 1
  }
  colnames(mat) <- uniq_val
  if(length(names(y)) > 0) rownames(mat) <- names(y)
  
  mat
}
