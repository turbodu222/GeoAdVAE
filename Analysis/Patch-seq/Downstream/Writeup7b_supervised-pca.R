rm(list=ls())

library(tidyverse)

source("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/code/kevin/Writeup7b_supervised-pca/color_key.R")
source("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/code/kevin/Writeup7b_supervised-pca/supervised_pca.R")

rna_df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/scala/exon_data_top2000_name.csv")
rownames(rna_df) <- rna_df[,"Cell"]
rna_df <- rna_df[,-1]
tmp <- stats::prcomp(rna_df)
rna_pca <- tmp$x[,1:50]

cajal_dist <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/scala/gw_dist.csv")
cajal_dist <- cajal_dist[,-1]
rownames(cajal_dist) <- rownames(rna_df)
tmp <- stats::prcomp(cajal_dist)
cajal_pca <- tmp$x[,1:50]

metadata <- read.table("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/scala/m1_patchseq_meta_data.csv", 
                       sep="\t", 
                       header=TRUE, 
                       check.names=FALSE)
rownames(metadata) <- metadata[,"Cell"]
metadata <- metadata[rownames(rna_pca),]

##########

# Check leading PCs first
color_vec <- cols_gex[metadata[,"RNA family"]]
plot(rna_pca[,1], rna_pca[,3], col = color_vec, pch = 16)

plot(cajal_pca[,1], cajal_pca[,2], col = color_vec, pch = 16)

##########

label_vec <- factor(plyr::mapvalues(x = metadata[,"RNA family"],
                                    from = c("CT", "IT", "ET", "NP",
                                             "Lamp5", "Sncg", "Vip", "Sst", "Pvalb",
                                             "low quality"),
                                    to = c(rep("E",4),
                                           rep("I",5),
                                           "low quality")))
table(metadata[,"RNA family"], label_vec)
label_mat <- as.matrix(model.matrix(~ label_vec - 1))

rna_spca <- supervised_pca(x = rna_pca,
                           y = label_mat,
                           k = 1)$dimred
plot(rna_spca[,1], col = color_vec, pch = 16)

cajal_spca <- supervised_pca(x = cajal_pca,
                             y = label_mat,
                             k = 1)$dimred
plot(cajal_spca[,1], col = color_vec, pch = 16)

############

plot(rna_spca[,1], cajal_spca[,1], col = color_vec, pch = 16)

############

plot_df <- data.frame(
  Cell = rownames(rna_spca),
  rna = rna_spca[,1],
  cajal = cajal_spca[,1],
  celltype = metadata[,"RNA family"],
  color = color_vec
)

library(ggplot2)

# 1. Recenter and scale the data (Mean = 0, SD = 1)
plot_df$rna_z <- as.numeric(scale(plot_df$rna))
plot_df$cajal_z <- as.numeric(scale(plot_df$cajal))

# 2. Compute the absolute distance matrix between all RNA and CAJAL points
# Rows = RNA points, Columns = CAJAL points
dist_mat <- abs(outer(plot_df$rna_z, plot_df$cajal_z, "-"))

# 3. RNA -> CAJAL Accuracy
# For each RNA value (row), find the index of the closest CAJAL value (column)
nn_cajal_idx <- apply(dist_mat, 1, which.min)
rna_to_cajal_acc <- mean(plot_df$celltype == plot_df$celltype[nn_cajal_idx])

# 4. CAJAL -> RNA Accuracy
# For each CAJAL value (column), find the index of the closest RNA value (row)
nn_rna_idx <- apply(dist_mat, 2, which.min)
cajal_to_rna_acc <- mean(plot_df$celltype == plot_df$celltype[nn_rna_idx])

# 5. Grand Mean
grand_mean <- (rna_to_cajal_acc + cajal_to_rna_acc) / 2

# Report Results
cat(sprintf("RNA to CAJAL 1-kNN Accuracy:  %.2f%%\n", rna_to_cajal_acc * 100))
cat(sprintf("CAJAL to RNA 1-kNN Accuracy:  %.2f%%\n", cajal_to_rna_acc * 100))
cat(sprintf("Grand Mean Accuracy:          %.2f%%\n", grand_mean * 100))

########################

# 2. Create a mapping for the legend using the colors provided in your table
# This ensures that 'ET' and 'IT' get the exact hex codes from your data
color_map <- unique(plot_df[, c("celltype", "color")])
cell_colors <- setNames(color_map$color, color_map$celltype)

# 3. Generate the plot
plot1 <- ggplot(plot_df) +
  # Draw the connecting gray lines first (so they stay behind the points)
  geom_segment(aes(x = 1, xend = 2, y = rna_z, yend = cajal_z), 
               color = "gray80", alpha = 0.4, linewidth = 0.3) +
  
  # Plot RNA points at x = 1
  geom_point(aes(x = 1, y = rna_z, color = celltype), size = 2.5) +
  
  # Plot CAJAL points at x = 2
  geom_point(aes(x = 2, y = cajal_z, color = celltype), size = 2.5) +
  
  # Use the manual colors and clean up the axes
  scale_color_manual(values = cell_colors) +
  scale_x_continuous(breaks = c(1, 2), 
                     labels = c("RNA\n(Supervised PCA)", "Morphology (CAJAL)\n(Supervised PCA)"),
                     limits = c(0.8, 2.2)) +
  labs(y = "Standardized 1D Embedding (Z-score)",
       x = NULL,
       color = "Cell Type",
       title = paste0("Accuracy: ", round(grand_mean*100,2), "%")) +
  theme_classic() +
  theme(
    axis.line.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid = element_blank()
  )

ggplot2::ggsave("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup7b/Writeup7b_supervised-pca.png",
                height = 3.5, width = 5)
