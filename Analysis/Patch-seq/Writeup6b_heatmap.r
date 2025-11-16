rm(list=ls())

library(Seurat)

plot_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup6b/"

load("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/kevin/Writeup6b/turbo_seurat_obj.RData")

gene_vec <- c("Gm20388",
              "mt.Nd1",
              "Malat1", 
              "mt.Co1",
              "Meg3", 
              "Hspa8", 
              "Actb",
              "Cst3")

cell_ids <- c("20190917_sample_3",
              "20190606_sample_3",
              "20180123_sample_5",
              "20190828_sample_5")

data_mat_subset <- SeuratObject::LayerData(
  seurat_obj,
  layer = "data",
  assay = "RNA",
  features = gene_vec
)
data_mat_subset <- as.matrix(data_mat_subset[gene_vec,])

for(j in 1:nrow(data_mat_subset)){
  min_val <- stats::quantile(data_mat_subset[j,], probs = 0.05)
  max_val <- stats::quantile(data_mat_subset[j,], probs = 0.95)
  data_mat_subset[j,] <- (data_mat_subset[j,]-min_val)/(max_val-min_val)
  data_mat_subset[j,] <- pmin(pmax(data_mat_subset[j,], 0), 1)
}

data_mat_subset <- data_mat_subset[,cell_ids]

num_cols <- 20
break_vec <- seq(min(data_mat_subset), 
                 max(data_mat_subset), 
                 length.out = num_cols+1)
break_vec[1] <- -1
break_vec[length(break_vec)] <- 2

png(paste0(plot_folder, "Writeup6b_gex_gene-exp.png"),
    height = 800, width = 1200, res = 300, units = "px")
par(mar = rep(0.5, 4))
image(data_mat_subset,
      col = grDevices::hcl.colors(num_cols, palette = "Berlin"),
      breaks = break_vec,
      xlab = "",
      ylab = "",
      bty = "n")
graphics.off()
