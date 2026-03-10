rm(list=ls())

folder_path <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/writeup22/Transport_Matrix/"
prefix <- "transport_matrix_batch_"

source("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/code/kevin/Writeup7b_supervised-pca/color_key.R")

fig_path <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup7e/"

.rotate <- function(a) { t(a[nrow(a):1,]) } 
.breakpoints <- function(membership_vec) {
  membership_vec <- as.numeric(membership_vec) ## convert to integers
  row_ordering <- order(membership_vec, decreasing = F)
  1-which(abs(diff(sort(membership_vec, decreasing = F))) >= 1e-6)/length(membership_vec)
}

#####

rna_df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/scala/exon_data_top2000_name.csv")
rownames(rna_df) <- rna_df[,"Cell"]
rna_df <- rna_df[,-1]

metadata <- read.table("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/scala/m1_patchseq_meta_data.csv",
                       sep="\t",
                       header=TRUE,
                       check.names=FALSE)
rownames(metadata) <- metadata[,"Cell"]
metadata <- metadata[rownames(rna_df),]

umap_df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/patchseq_onlyGEX_umap_GEX.csv",
                    row.names = 1)
umap_df <- umap_df[rownames(rna_df),]

####

library(vroom)
library(future.apply)

# 1. Setup Parallel Processing (uses all but one of your CPU cores)
plan(multisession, workers = parallel::detectCores() - 1)

# 2. Pre-generate file paths
file_paths <- sprintf("%s%s%04d.csv", folder_path, prefix, 0:499)

# 3. Parallelized, optimized read
transport_list <- future_lapply(file_paths, function(path) {
  print(path)
  
  # vroom is significantly faster than read.csv
  # col_select = -1 handles removing the first column during the read itself
  df <- vroom(path, col_select = -1, show_col_types = FALSE)
  
  # Convert to matrix directly
  mat <- as.matrix(df)
  
  # Clean column names (strip "X")
  colnames(mat) <- gsub("X", "", colnames(mat))
  rownames(mat) <- colnames(mat)
  
  # Vectorized math operation
  round(exp(mat) * nrow(mat), 1)
})

#########

n <- nrow(metadata)
frequency_mat <- matrix(0, nrow = n, ncol = n, dimnames = 
                          list(rownames(metadata), 
                               rownames(metadata)))
for(iter in 1:length(transport_list)){
  df <- transport_list[[iter]]
  
  for(i in 1:nrow(df)){
    cell_from <- rownames(df)[i]
    cells_to <- colnames(df)[which(df[cell_from,]>0)]
    frequency_mat[cell_from,cells_to] <- frequency_mat[cell_from,cells_to] + df[cell_from,cells_to]
  }
}

#########

svd_res <- svd(frequency_mat)
# plot(svd_res$d); lines(x = rep(10,2), y = c(-1e4,1e4), col = 2, lwd = 2)

set.seed(10)
k <- 10
row_clust <- stats::kmeans(svd_res$u[,1:k], centers = k)
col_clust <- stats::kmeans(svd_res$v[,1:k], centers = k)

# reassign the singleton
.remove_singletons <- function(vec){
  tab_vec <- table(vec)
  tab_vec <- sort(tab_vec, decreasing = TRUE)
  if(any(tab_vec) == 1){
    singletons <- names(tab_vec)[tab_vec == 1]
    for(singleton in singletons){
      idx <- which(vec == singleton)
      vec[idx] <- names(tab_vec)[1]
    }
  }
  
  as.numeric(droplevels(factor(vec)))
}

row_clust$cluster <- .remove_singletons(row_clust$cluster)
col_clust$cluster <- .remove_singletons(col_clust$cluster)

k <- length(unique(row_clust$cluster))

###
# Reorder the rows and columns. First the rows
tab <- table(row_clust$cluster)
new_order <- names(sort(tab, decreasing = FALSE))
row_clust$cluster <- match(as.character(row_clust$cluster), new_order)

block_avg <- matrix(0, nrow = k, ncol = k, dimnames = list(paste0("r", 1:k), paste0("c", 1:k)))
for(i in 1:k) {
  for(j in 1:k) {
    # Extract the sub-matrix for block (i, j)
    sub_mat <- frequency_mat[row_clust$cluster == i, col_clust$cluster == j]
    block_avg[i, j] <- mean(sub_mat)
  }
}

new_col_order <- c(); matching_row_order <- c()
block_tmp <- block_avg
while(length(new_col_order) < k){
  idx <- which(block_tmp == max(block_tmp), arr.ind = TRUE)
  new_col_order <- c(new_col_order, colnames(block_tmp)[idx[1,2]])
  matching_row_order <- c(matching_row_order, rownames(block_tmp)[idx[1,1]])
  block_tmp <- block_tmp[-idx[1,1],-idx[1,2], drop = FALSE]
}
# round(block_avg[matching_row_order,new_col_order]*100)
names(new_col_order) <- matching_row_order
new_col_order <- new_col_order[paste0("r", 1:k)]
new_col_order <- as.numeric(gsub(pattern = "c", replacement = "", new_col_order))
col_clust$cluster <- match(as.character(col_clust$cluster), new_col_order)

###

metadata <- read.table("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/scala/m1_patchseq_meta_data.csv",
                       sep="\t",
                       header=TRUE,
                       check.names=FALSE)
rownames(metadata) <- metadata[,"Cell"]

family_levels <- c("CT", "ET", "IT", "NP", "Lamp5", "Pvalb", "Sncg", "Sst", "Vip", "low quality")

# Convert the metadata column to a factor with these levels
# This ensures that order() respects your custom sequence
metadata[,"RNA family"] <- factor(metadata[,"RNA family"], levels = family_levels)
# Generate the new order
# This sorts by cluster (1 to 10) first, and then by the RNA family factor second
metadata <- metadata[rownames(frequency_mat),]
row_order <- order(row_clust$cluster, metadata[,"RNA family"])

metadata <- metadata[colnames(frequency_mat),]
col_order <- unlist(lapply(1:k, function(j){
  idx <- which(col_clust$cluster == j)
  idx[order(metadata[idx,"RNA family"])]
}))

frequency_mat2 <- frequency_mat[row_order, col_order]

frequency_mat2[frequency_mat2 >= 2] <- 2

row_breaks <- .breakpoints(row_clust$cluster)
col_breaks <- .breakpoints(col_clust$cluster)

png(paste0(fig_path, "Writeup7e_adjacency.png"),
    height = 1200, width = 1200, units = "px", res = 300)
par(mar = rep(0.5, 4))
image(.rotate(frequency_mat2), asp = TRUE,
      breaks = c(-0.5, 0.5, 1.5, 2.5),
      col = c("white", "black", "black"),
      main = "",
      xaxt = "n",
      yaxt = "n",
      xlab = "",
      ylab = "")
abline(h = row_breaks[-k], v = 1-col_breaks[-k], col = "black", lwd = 1.5, lty = 2)
graphics.off()

#######

project_doubly_stochastic <- NA
project_doubly_stochastic <- function(M, iter = 100, tol = 1e-8) {
  # M must be non-negative
  M <- as.matrix(M)
  
  for (i in 1:iter) {
    # 1. Row normalization
    row_sums <- rowSums(M)
    row_sums[row_sums == 0] <- 1 # Avoid division by zero
    M <- M / row_sums
    
    # Check for convergence after row step (optional)
    col_sums <- colSums(M)
    if (all(abs(col_sums - 1) < tol)) break
    
    # 2. Column normalization
    M <- t(t(M) / col_sums)
  }
  return(M)
}

# Usage
boolean_mat <- frequency_mat
boolean_mat[boolean_mat!=0] <- 1
block_avg <- matrix(0, nrow = k, ncol = k)

for(i in 1:k) {
  for(j in 1:k) {
    # Extract the sub-matrix for block (i, j)
    sub_mat <- boolean_mat[row_clust$cluster == i, col_clust$cluster == j]
    block_avg[i, j] <- mean(sub_mat)
  }
}

block_avg <- project_doubly_stochastic(block_avg)


plot_data <- .rotate(block_avg)

# Plot the summary heatmap
png(paste0(fig_path, "Writeup7e_block-avg.png"),
    height = 1200, width = 1200, units = "px", res = 300)
par(mar = rep(0.5, 4))
image(plot_data, 
      asp = TRUE,
      col = hcl.colors(12, "YlOrRd", rev = TRUE),
      main = "",
      xaxt = "n",
      yaxt = "n",
      xlab = "",
      ylab = "")

# 3. Define the coordinates for the centers of the k x k blocks
# In image(), the centers are spread evenly from 0 to 1
cell_centers <- seq(0, 1, length.out = k)
grid_coords <- expand.grid(x = cell_centers, y = cell_centers)

# 4. Add the values as text
# We round to 2 decimal places and format as a string
labels <- sprintf("%.2f", plot_data)

text(grid_coords$x, grid_coords$y, 
     labels = labels, 
     cex = 0.6,          # Adjust size based on k
     col = ifelse(plot_data > 0.4, "white", "black")) # Contrast logic

graphics.off()

#############

png(paste0(fig_path, "Writeup7e_row-order.png"),
    height = 300, width = 1200, units = "px", res = 300)
par(mar = rep(0.5, 4))
metadata <- metadata[rownames(frequency_mat2),]
row_num <- as.numeric(metadata[,"RNA family"])
tmp_mat <- matrix(row_num, ncol = 1)
image(tmp_mat, 
      breaks = seq(0.5, length(family_levels)+1, by = 1),
      col = cols_gex[family_levels],
      main = "",
      xaxt = "n",
      yaxt = "n",
      xlab = "",
      ylab = "")
graphics.off()
      

png(paste0(fig_path, "Writeup7e_col-order.png"),
    height = 300, width = 1200, units = "px", res = 300)
par(mar = rep(0.5, 4))
metadata <- metadata[colnames(frequency_mat2),]
col_num <- as.numeric(metadata[,"RNA family"])
tmp_mat <- matrix(col_num, ncol = 1)
image(tmp_mat, 
      breaks = seq(0.5, length(family_levels)+1, by = 1),
      col = cols_morph[family_levels],
      main = "",
      xaxt = "n",
      yaxt = "n",
      xlab = "",
      ylab = "")
graphics.off()

#############

library(ggplot2)

# 1. Calculate percentage of variance explained
# Variance is proportional to the square of the singular values
var_explained <- (svd_res$d^2) / sum(svd_res$d^2) * 100
df_plot <- data.frame(
  Principal_Component = 1:length(var_explained),
  Variance = var_explained
)

# 2. Create the publication-ready plot
plot1 <- ggplot(df_plot[1:100, ], aes(x = Principal_Component, y = Variance)) +
  # Add a subtle area fill to emphasize the curve
  geom_area(fill = "steelblue", alpha = 0.2) +
  # Add the line and points
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 1.5) +
  # Draw the elbow line at k=10
  geom_vline(xintercept = 9, linetype = "dashed", color = "firebrick", linewidth = 0.8) +
  # Annotate the elbow
  annotate("text", x = 12, y = max(var_explained)*0.8, 
           label = "k = 9 (Elbow)", color = "firebrick", hjust = 0, fontface = "bold") +
  # Clean, professional theme
  theme_classic(base_size = 14) +
  labs(
    title = "SVD Scree Plot",
    subtitle = "Singular value spectrum with k=10 heuristic",
    x = "Singular Value Rank",
    y = "% Variance Explained"
  ) +
  # Force the plot to start at 0
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  scale_x_continuous(breaks = c(1, seq(10, 100, by = 10)))


# Save for publication
ggplot2::ggsave(paste0(fig_path, "Writeup7e_SVD_scree.pdf"),
       plot1, width = 6, height = 4, device = "pdf")
