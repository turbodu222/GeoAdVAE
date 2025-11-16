rm(list=ls())

library(Seurat)
library(princurve)

plot_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup6/"

gex_mat <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/5xFAD_GeoAdvAE_GEX_latent_coor_5XFAD.csv",
                    row.names = 1)
morph_mat <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/5xFAD_GeoAdvAE_Morpho_latent_coor_5XFAD.csv")
rownames(morph_mat) <- morph_mat$X0

gex_df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/5xFAD_GeoAdvAE_GEX_UMAP_coor_5XFAD.csv",
                   row.names = 1)
morph_df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/5xFAD_GeoAdvAE_Morpho_UMAP_coor_5XFAD.csv")
rownames(morph_df) <- morph_df$X0

gex_df$cluster_label <- paste0("gex_", as.character(gex_df$cluster_label))
morph_df$cluster_label <- paste0("morph_", as.character(morph_df$cluster_label))
morph_df <- morph_df[,c("umap_1", "umap_2", "cluster_label", "modality")]

total_mat <- rbind(gex_mat, morph_mat[paste0("latent_dim_", 1:15)])
colnames(total_mat) <- paste0("dummy-", 1:ncol(total_mat))
total_mat <- as.matrix(total_mat)
metadata <- rbind(gex_df, morph_df)
colnames(metadata) <- c("turbo_umap_1", "turbo_umap_2", "cluster_label", "modality")
metadata <- metadata[rownames(total_mat),]

seurat_obj <- Seurat::CreateSeuratObject(counts = t(total_mat), 
                                         meta.data = metadata)
Seurat::VariableFeatures(seurat_obj) <- colnames(total_mat)

colnames(total_mat) <- paste0("latent-dim-", 1:ncol(total_mat))
seurat_obj[["pca"]] <- Seurat::CreateDimReducObject(embeddings = total_mat,
                                                    key = "pca_")

K <- ncol(seurat_obj[["pca"]]@cell.embeddings)
set.seed(10)
seurat_obj <- Seurat::RunUMAP(seurat_obj, 
                              dims = 1:K)

col_palette <- c(
  "gex_0"="#0B559F", "gex_1"="#565AD6", "gex_2"="#CFE8FF", "gex_4"="#8FC2F0", "gex_5"="#074ED3",
  "gex_7"="#FFCF70", "gex_3"="#F39C34", "gex_8"="#D04E00",
  "gex_10"="#A7E08B", "gex_11"="#2E9F51", "gex_9"="#338538",
  "gex_6"="#DB72F5",
  "morph_0" = "#FF2F92",
  "morph_1" = "#00E1FF",
  "morph_2" = "#8C564B"
)

Seurat::DimPlot(seurat_obj, 
                group.by = "cluster_label",
                cols = col_palette)

##########################

set.seed(10)
seurat_obj <- Seurat::FindNeighbors(seurat_obj, dims = 1:K)
seurat_obj <- Seurat::FindClusters(seurat_obj, resolution = 0.1)

Seurat::DimPlot(seurat_obj, 
                group.by = "seurat_clusters")

.initial_curve_fit <- function(cluster_vec,
                               dimred, 
                               lineage_order){
  stopifnot(all(lineage_order %in% cluster_vec),
            length(cluster_vec) == nrow(dimred),
            is.factor(cluster_vec))
  t(sapply(lineage_order, function(cluster){
    idx <- which(cluster_vec == cluster)
    Matrix::colMeans(dimred[idx,,drop = F])
  }))
}

.extract_pseudotime <- function(dimred,
                                initial_fit,
                                stretch){ # default stretch=2
  pcurve <- princurve::project_to_curve(dimred,
                                        s = initial_fit,
                                        stretch = stretch)
  pcurve$lambda
}

#########################

cluster_vec <- as.character(seurat_obj$RNA_snn_res.0.1)
dimred <- seurat_obj[["pca"]]@cell.embeddings
umap_mat <- seurat_obj[["umap"]]@cell.embeddings

idx <- which(cluster_vec == "7")
cluster_vec <- cluster_vec[-idx]
dimred <- dimred[-idx,]
umap_mat <- umap_mat[-idx,]

initial_fit <- .initial_curve_fit(cluster_vec = factor(cluster_vec),
                          dimred = dimred,
                          lineage_order = c("6","5","4","3","1","2","0"))
slingshot_res <- .extract_pseudotime(dimred = dimred,
                                     initial_fit = initial_fit,
                                     stretch = 2)

time_vec <- rep(NA, length(Seurat::Cells(seurat_obj)))
names(time_vec) <- Seurat::Cells(seurat_obj)
time_vec[names(slingshot_res)] <- slingshot_res
seurat_obj$pseudotime <- time_vec
Seurat::FeaturePlot(seurat_obj, features = "pseudotime")

color_palette <- grDevices::colorRampPalette(c("darkgreen", "beige"))(50)
value_palette <- seq(min(slingshot_res), max(slingshot_res), length.out = 50)
col_vec <- sapply(slingshot_res, function(x){
  color_palette[which.min(abs(x - value_palette))]
})
plot(umap_mat,
     pch = 16,
     col = col_vec)


write.csv(time_vec,
          file = "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6/Writeup6_time.csv")

