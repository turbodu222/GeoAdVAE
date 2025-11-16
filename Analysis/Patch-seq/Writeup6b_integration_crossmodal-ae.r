rm(list=ls())

plot_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup6b/"

df_gex <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/patchseq_crossmodalAE_umap_gene_expression_modality_b.csv",
                   row.names = 1)
df_morph <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/patchseq_crossmodalAE_umap_morphology_modality_a.csv",
                     row.names = 1)

metadata <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/data/scala/m1_patchseq_meta_data.csv", 
                     sep = "\t")

all(rownames(df_gex) %in% metadata$Cell)

RNA.family_vec <- metadata$RNA.family
names(RNA.family_vec) <- metadata$Cell
df_gex$celltype <- RNA.family_vec[rownames(df_gex)]
df_morph$celltype <- RNA.family_vec[rownames(df_morph)]

cluster_cols_gex <- c(
  "IT"         = "#ED3535",
  "ET"         = "#F47506",
  "CT"         = "#DF659C",
  "NP"         = "#C5A34C",
  "Pvalb"      = "#2C7CD3",
  "Sst"        = "#51D4A2",
  "Vip"        = "#6A3FF6",
  "Lamp5"      = "#26DCF4",
  "Sncg"       = "#C063FA",
  "low quality"= "#6A6A6A"
)

cluster_cols_morph <- c(
  "IT"       = "#F29E9E",
  "ET"       = "#F6B48F",
  "CT"       = "#E6A0C4",
  "NP"       = "#F0C0A0",
  "Pvalb"    = "#A8C7E0",
  "Sst"      = "#A9D6CB",
  "Vip"      = "#B7B5F0",
  "Lamp5"    = "#9FD6E3",
  "Sncg"     = "#C9B8E6",
  "low quality" = "#A8A8A8"
)

umap_gex <- as.matrix(df_gex[,c("UMAP1", "UMAP2")])
umap_morph <- as.matrix(df_morph[,c("UMAP1", "UMAP2")])
umap_morph <- umap_morph[rownames(umap_gex),]


png(paste0(plot_folder, "Writeup6b_umap-integration_crossmodalAE.png"),
    height = 800, width = 1200, res = 300, units = "px")

par(mar = rep(0.5, 4))

set.seed(10); shuf_idx <- sample(1:nrow(umap_gex))
plot(x = umap_gex[shuf_idx,1], 
     y = umap_gex[shuf_idx,2],
     pch = 16,
     col = cluster_cols_gex[as.character(df_gex$celltype)][shuf_idx],
     xlim = range(c(umap_gex[,1], umap_morph[,1])),
     ylim = range(c(umap_gex[,2], umap_morph[,2])),
     xaxt = "n",
     yaxt = "n",
     bty = "n",
     xlab = "",
     ylab = "")

# draw lines
for(i in 1:nrow(umap_morph)){
  j <- which(rownames(umap_gex) == rownames(umap_morph)[i])
  lines(x = c(umap_gex[j,1], umap_morph[i,1]),
        y = c(umap_gex[j,2], umap_morph[i,2]),
        lwd = 0.5,
        col = rgb(0.5, 0.5, 0.5, 0.5))
}

set.seed(10); shuf_idx2 <- sample(1:nrow(umap_morph))
points(x = umap_morph[shuf_idx2,1], 
       y = umap_morph[shuf_idx2,2],
       pch = 17,
       col = "black",
       cex = 1.5)

points(x = umap_morph[shuf_idx2,1], 
       y = umap_morph[shuf_idx2,2],
       pch = 17,
       col = "white",
       cex = 1.25)

points(x = umap_morph[shuf_idx2,1], 
       y = umap_morph[shuf_idx2,2],
       pch = 17,
       col = cluster_cols_morph[as.character(df_morph$celltype)][shuf_idx2])

graphics.off()
