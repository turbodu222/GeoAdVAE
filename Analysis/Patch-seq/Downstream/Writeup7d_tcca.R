rm(list=ls())

library(tidyverse)
library(Seurat)
library(tiltedCCA)

source("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/code/kevin/Writeup7b_supervised-pca/color_key.R")
source("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/code/kevin/Writeup7b_supervised-pca/supervised_pca.R")

rna_df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/scala/exon_data_top2000_name.csv")
rownames(rna_df) <- rna_df[,"Cell"]
rna_df <- rna_df[,-1]
tmp <- stats::prcomp(rna_df)
par(mfrow = c(1,2)); plot(tmp$sdev); plot(tmp$sdev[1:20]); par(mfrow = c(1,1))
rna_pca <- tmp$x[,1:50]

cajal_dist <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/scala/gw_dist.csv")
cajal_dist <- cajal_dist[,-1]
rownames(cajal_dist) <- rownames(rna_df)
tmp <- stats::prcomp(cajal_dist)
par(mfrow = c(1,2)); plot(tmp$sdev); plot(tmp$sdev[1:20]); par(mfrow = c(1,1))
cajal_pca <- tmp$x[,1:50]

metadata <- read.table("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/scala/m1_patchseq_meta_data.csv",
                       sep="\t",
                       header=TRUE,
                       check.names=FALSE)
rownames(metadata) <- metadata[,"Cell"]
metadata <- metadata[rownames(rna_pca),]


morph_df <- read.table("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/data/scala/m1_patchseq_morph_features.csv",
                       sep=",", 
                       header=TRUE, 
                       check.names=FALSE)
rownames(morph_df) <- morph_df[,"cell id"]
morph_df <- morph_df[rownames(rna_df),]

table(metadata[,"RNA family"], morph_df[,"cell class"])

###################

# label_vec <- factor(plyr::mapvalues(x = metadata[,"RNA family"],
#                                     from = c("CT", "IT", "ET", "NP",
#                                              "Lamp5", "Sncg", "Vip", "Sst", "Pvalb",
#                                              "low quality"),
#                                     to = c(rep("E",4),
#                                            rep("I",5),
#                                            "low quality")))

################

# https://linnykos.github.io/tiltedCCA/articles/bm.html
# https://linnykos.github.io/tiltedCCA/articles/simulation.html

set.seed(10)
multiSVD_obj <- tiltedCCA::create_multiSVD(mat_1 = rna_pca, mat_2 = cajal_pca,
                                           dims_1 = 1:50, dims_2 = 1:50,
                                           center_1 = T, center_2 = T,
                                           normalize_row = T,
                                           normalize_singular_value = T,
                                           recenter_1 = F, recenter_2 = F,
                                           rescale_1 = F, rescale_2 = F,
                                           scale_1 = T, scale_2 = T)
multiSVD_obj <- tiltedCCA::form_metacells(input_obj = multiSVD_obj,
                                          large_clustering_1 = NULL, 
                                          large_clustering_2 = NULL,
                                          num_metacells = NULL)
multiSVD_obj <- tiltedCCA::compute_snns(input_obj = multiSVD_obj,
                                        latent_k = 5,
                                        num_neigh = 20,
                                        bool_cosine = T,
                                        bool_intersect = F,
                                        min_deg = 5,
                                        verbose = 2)
multiSVD_obj <- tiltedCCA::tiltedCCA(input_obj = multiSVD_obj)
multiSVD_obj <- tiltedCCA::fine_tuning(input_obj = multiSVD_obj)
multiSVD_obj <- tiltedCCA::tiltedCCA_decomposition(multiSVD_obj)

##########################

tcca_mat <- multiSVD_obj$tcca_obj$common_score
tmp <- Seurat::RunUMAP(tcca_mat)
tcca_umap <- tmp@cell.embeddings

color_vec <- cols_gex[metadata[,"RNA family"]]

png(paste0("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup7d/Writeup7d_tcca.png"),
    height = 1500, width = 1500, units = "px", res = 300)
plot(x = tcca_umap[,1],
     y = tcca_umap[,2],
     col = color_vec,
     pch = 16,
     main = "",
     xlab = "UMAP 1 (of TCCA common space)",
     ylab = "UMAP 2 (of TCCA common space)")
graphics.off()

