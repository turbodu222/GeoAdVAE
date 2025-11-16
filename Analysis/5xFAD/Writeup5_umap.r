rm(list=ls())

library(Seurat)
out_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/kevin/Writeup5/"
data_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/data/GSE150358/"
plot_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup5/"

load(paste0(out_folder, "wang_microglia_cleaned.RData"))

cluster_colors <- c(
  "0"="#0B559F", "1"="#565AD6", "2"="#CFE8FF", "4"="#8FC2F0", "5"="#074ED3",
  "7"="#FFCF70", "3"="#F39C34", "8"="#D04E00",
  "10"="#A7E08B", "11"="#2E9F51", "9"="#338538",
  "6"="#DB72F5"
)
# In Seurat:
plot1 <- Seurat::DimPlot(seurat_obj, 
                         group.by = "microgliaClusterID", 
                         cols = cluster_colors)
ggplot2::ggsave(plot1, filename = paste0(plot_folder, "Writeup5_wang_umap-recolored.png"),
                height = 5, width = 5)

umap_mat <- seurat_obj[["umap"]]@cell.embeddings
celltype_vec <- as.character(seurat_obj$microgliaClusterID)
color_vec <- cluster_colors[celltype_vec]

set.seed(10)
shuff_idx <- sample(nrow(umap_mat))

png(paste0(plot_folder, "Writeup5_wang_umap-recolored_cleaned.png"),
    height = 800, width = 1200, res = 300, units = "px")
par(mar = rep(0.5,4))
plot(umap_mat[shuff_idx,],
     pch = 16,
     col = color_vec[shuff_idx],
     xaxt = "n",
     yaxt = "n",
     bty = "n",
     cex = 0.75)
graphics.off()

#######################################

gene_list = list(
  Homeostatic = c("Tmem119", "P2ry12", "Ivns1abp", "Selplg", "Stab1", "Cd164", "Crybb1", "Ank"),
  DAM2 = c("Spp1", "Axl", "Csf1", "Cst7", "Cd9"),
  DAM1 = c("Cadm1", "Apoe", "B2m", "Cstb", "Tyrobp", "Timp2", "Fth1"),
  G2M = c("Top2a", "Mki67", "Pbk", "Racgap1", "Tpx2", "Cenpe", "Cdca3"),
  S = c("Mcm5", "Pcna", "Tyms", "Fen1", "Mcm2", "Mcm4", "Rrm1"),
  IRM = c("Ifit2", "Ifit3", "Irf7", "Oasl2", "Ifit1", "Ifi209", "Ifi213")
)

table(unlist(gene_list) %in% SeuratObject::Features(seurat_obj))

seurat_obj <- Seurat::AddModuleScore(seurat_obj, 
                                     features = gene_list)
metadata <- seurat_obj@meta.data
colnames(metadata)[grep("^Cluster", colnames(metadata))] <- names(gene_list)
seurat_obj@meta.data <- metadata

celltype_conversion <- c("0"="Homeostatic1", 
                         "1"="Homeostatic2", 
                         "2"="Homeostatic3", 
                         "4"="Homeostatic4", 
                         "5"="Homeostatic5",
                         "7"="Transition", 
                         "3"="DAM2", 
                         "8"="DAM1",
                         "10"="Proliferative1", 
                         "11"="Proliferative2", 
                         "9"="Proliferative3",
                         "6"="IRM"
)

celltype_labels <- as.character(seurat_obj$microgliaClusterID)
celltype_labels <- plyr::mapvalues(x = celltype_labels,
                                   from = names(celltype_conversion),
                                   to = celltype_conversion)
seurat_obj$celltype <- celltype_labels

plot1 <- Seurat::DotPlot(seurat_obj, 
                features = names(gene_list), 
                group.by = "celltype")
ggplot2::ggsave(plot1, filename = paste0(plot_folder, "Writeup5_wang_modules.png"),
                height = 5, width = 9)

ggplot2::ggsave(plot1, filename = paste0(plot_folder, "Writeup5_wang_modules_resized.png"),
                height = 4, width = 6)
