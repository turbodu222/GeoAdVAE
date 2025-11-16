rm(list=ls())

library(Seurat)
out_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/kevin/Writeup5/"
plot_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup6/"

load(paste0(out_folder, "wang_microglia_cleaned.RData"))
time_vec_full <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6/Writeup6_time.csv",
                     row.names = 1)

time_vec <- time_vec_full[Seurat::Cells(seurat_obj),1]

seurat_obj$time <- time_vec

mat <- SeuratObject::LayerData(seurat_obj, 
                               layer = "data",
                               assay = "RNA",
                               features = Seurat::VariableFeatures(seurat_obj))

nonna_time <- which(!is.na(time_vec))

corr_mat <- sapply(1:nrow(mat), function(j){
  if(j %% floor(nrow(mat)/10) == 0) cat('*')
  
  vec <- as.numeric(mat[j,])
  nonzero_idx <- intersect(which(vec != 0), nonna_time)
  
  if(length(nonzero_idx) > 0){
    val <- stats::cor(vec[nonzero_idx], time_vec[nonzero_idx], use = "complete.obs")
    num_nonzero <- length(nonzero_idx)
  } else{
    val <- NA
    num_nonzero <- 0
  }
  
  c(value = val, num_nonzero = num_nonzero)
})
colnames(corr_mat) <- rownames(mat)
corr_mat <- t(corr_mat)

plot(corr_mat[,1], corr_mat[,2])

corr_mat <- corr_mat[which(corr_vec[,2] >= 10000),]

###############

idx <- which.min(corr_mat[,1])
gene_name <- rownames(corr_mat)[idx]
# idx <- which(corr_mat <= -0.1)

# plot(mat[gene_name, ], 
#      time_vec,
#      pch = 16,
#      col = rgb(0.5, 0.5, 0.5, 0.1))

######################

# plot microglia gex states by pseudotime

mat <- matrix(NA, nrow = nrow(time_vec_full), ncol = 2)
mat[,2] <- time_vec_full[,1]
rownames(mat) <- rownames(time_vec_full)
mat[Seurat::Cells(seurat_obj),1] <- seurat_obj$microgliaClusterID
mat[which(is.na(mat[,1])),1] <- -1
mat <- mat[which(!is.na(mat[,2])),]
mat <- mat[order(mat[,2]),]

gex_palette <- c(
  "0"="#0B559F", "1"="#565AD6", "2"="#CFE8FF", 
  "3"="#F39C34", 
  "4"="#8FC2F0", "5"="#074ED3",
  "6"="#DB72F5", "7"="#FFCF70", "8"="#D04E00",
  "9"="#338538", "10"="#A7E08B", "11"="#2E9F51"
)

png(paste0(plot_folder, "Writeup6_gex_pseudotime.png"),
    height = 800, width = 1200, res = 300, units = "px")
par(mar = rep(0.5, 4))
image(mat[,1,drop = FALSE],
      col = c("white", gex_palette),
      breaks = seq(-1.5,11.5, by = 1),
      xlab = "",
      ylab = "",
      bty = "n")
graphics.off()

######################

# plot microglia morph states by pseudotime
morph_df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/5xFAD_GeoAdvAE_Morpho_UMAP_coor_5XFAD.csv",
                     row.names = 1)

mat <- matrix(NA, nrow = nrow(time_vec_full), ncol = 2)
mat[,2] <- time_vec_full[,1]
rownames(mat) <- rownames(time_vec_full)
mat[morph_df$X0,1] <- morph_df$cluster_label
mat[which(is.na(mat[,1])),1] <- -1
mat <- mat[which(!is.na(mat[,2])),]
mat <- mat[order(mat[,2]),]

# make the morph cells "thicker" by 100x
idx <- which(mat[,1] != -1)
for(i in idx){
  fill_vec <- setdiff(max(i-50,1):min(i+50,nrow(mat)), idx)
  mat[fill_vec,1] <- mat[i,1]
}

morph_palette <- c(
  "0" = "#FF2F92",
  "1" = "#00E1FF",
  "2" = "#8C564B"
)

png(paste0(plot_folder, "Writeup6_morph_pseudotime.png"),
    height = 800, width = 1200, res = 300, units = "px")
par(mar = rep(0.5, 4))
image(mat[,1,drop = FALSE],
      col = c("white", morph_palette),
      breaks = seq(-1.5,2.5, by = 1),
      xlab = "",
      ylab = "",
      bty = "n")
graphics.off()

########################

corr_mat <- corr_mat[order(corr_mat[,"value"]),]

rownames(corr_mat)[which(corr_mat[,"value"] <= -0.5)]
rownames(corr_mat)[which(corr_mat[,"value"] >= 0.2)]

genes <- c("Acsl1", "S100a9", "Ms4a6b",
           "Ftl1", "Tyrobp", "Fth1")

corr_mat[genes,]

expr_mat <- SeuratObject::LayerData(seurat_obj, 
                               layer = "data",
                               assay = "RNA",
                               features = genes)
expr_mat <- expr_mat[genes,]

mat <- matrix(NA, nrow = nrow(time_vec_full), ncol = length(genes)+1)
mat[,ncol(mat)] <- time_vec_full[,1]
rownames(mat) <- rownames(time_vec_full)
mat[Seurat::Cells(seurat_obj),1:length(genes)] <- as.matrix(t(expr_mat))
mat <- mat[which(!is.na(mat[,ncol(mat)])),]
mat <- mat[order(mat[,ncol(mat)]),]

for(j in 1:length(genes)){
  vec <- mat[,j]
  idx <- which(vec >= 1e-6)
  vec <- vec[idx]
  max_val <- stats::quantile(vec, probs = 0.9)
  min_val <- stats::quantile(vec, probs = 0.1)
  mat[idx,j] <- pmin(mat[idx,j], max_val)
  mat[idx,j] <- pmax(mat[idx,j], min_val)
  mat[idx,j] <- (mat[idx,j]-min_val)/(max_val-min_val)+1e-6
  mat[setdiff(1:nrow(mat), idx),j] <- -1
}

num_cols <- 20
break_vec <- c(-1.5, seq(0, 1, length.out = num_cols+1))
break_vec[length(break_vec)] <- 1.1

png(paste0(plot_folder, "Writeup6_gex_gene-exp.png"),
    height = 800, width = 1200, res = 300, units = "px")
par(mar = rep(0.5, 4))
image(mat[,1:length(genes),drop = FALSE],
      col = c("white", grDevices::hcl.colors(num_cols, palette = "Berlin")),
      breaks = break_vec,
      xlab = "",
      ylab = "",
      bty = "n")
graphics.off()

# plot(1:num_cols, col = grDevices::hcl.colors(num_cols, palette = "Berlin"), pch = 16, cex = 5)
