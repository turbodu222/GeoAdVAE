rm(list=ls())

plot_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup6b/"

df_gex <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/patchseq_onlyGEX_umap_GEX.csv",
                   row.names = 1)
df_morph <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/patchseq_onlyMorph_umap_morpho_cajal.csv",
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

umap_gex <- as.matrix(df_gex[,c("UMAP_1", "UMAP_2")])
umap_morph <- as.matrix(df_morph[,c("UMAP_1", "UMAP_2")])

set.seed(10); shuf_idx <- sample(1:nrow(umap_gex))
plot(x = umap_gex[shuf_idx,1], 
     y = umap_gex[shuf_idx,2],
     pch = 16,
     col = cluster_cols_gex[as.character(df_gex$celltype)][shuf_idx])

set.seed(10); shuf_idx2 <- sample(1:nrow(umap_morph))
plot(x = umap_morph[shuf_idx2,1], 
     y = umap_morph[shuf_idx2,2],
     pch = 16,
     col = cluster_cols_morph[as.character(df_morph$celltype)][shuf_idx2])


# do a simple alignment

.rotate_matrix <- function(source_mat, 
                           target_mat){
  stopifnot(all(dim(source_mat) == dim(target_mat)))
  
  tmp <- svd(t(source_mat) %*% target_mat)
  rotation_mat <- tmp$u %*% t(tmp$v)
  tmp <- target_mat %*% t(rotation_mat)
  rownames(tmp) <- rownames(target_mat)
  colnames(tmp) <- colnames(target_mat)
  tmp
}

umap_morph <- .rotate_matrix(source_mat = umap_gex[rownames(umap_morph),],
                             target_mat = umap_morph)

plot(x = umap_morph[shuf_idx2,1], 
     y = umap_morph[shuf_idx2,2],
     pch = 16,
     col = cluster_cols_morph[as.character(df_morph$celltype)][shuf_idx2])


####################################

# first, renormalize the umap coords
umap_gex <- apply(umap_gex, 2, function(x){
  (x-min(x))/(max(x)-min(x)) - .5
})
umap_gex[,2] <- 2*umap_gex[,2]
umap_morph <- apply(umap_morph, 2, function(x){
  (x-min(x))/(max(x)-min(x)) - .5
})
umap_morph[,2] <- 2*umap_morph[,2]

umap_morph[,1] <- umap_morph[,1]+1.25
umap_morph[,2] <- 0.8*umap_morph[,2]

png(paste0(plot_folder, "Writeup6b_umap-alignment.png"), 
    height = 1000, width = 2500,
    units = "px", res = 300)
par(mar = rep(0.5, 4))
set.seed(10); shuf_idx2 <- sample(1:nrow(umap_morph))
plot(x = umap_morph[shuf_idx2,1], 
     y = umap_morph[shuf_idx2,2],
     pch = 16,
     col = cluster_cols_morph[as.character(df_morph$celltype)][shuf_idx2],
     xaxt = "n",
     yaxt = "n",
     bty = "n",
     xlim = c(-.5,1.75),
     ylim = c(-1,1),
     xlab = "",
     ylab = "")

# draw lines
for(i in 1:nrow(umap_morph)){
  j <- which(rownames(umap_gex) == rownames(umap_morph)[i])
  lines(x = c(umap_gex[j,1], umap_morph[i,1]),
        y = c(umap_gex[j,2], umap_morph[i,2]),
        lwd = 0.5,
        col = rgb(0.5, 0.5, 0.5, 0.2))
}

# put morphology
set.seed(10); shuf_idx <- sample(1:nrow(umap_gex))

points(x = umap_gex[shuf_idx,1], 
       y = umap_gex[shuf_idx,2],
       pch = 16,
       col = cluster_cols_gex[as.character(df_gex$celltype)][shuf_idx])
graphics.off()

##########


png(paste0(plot_folder, "Writeup6b_umap-alignment_no-lines.png"), 
    height = 1000, width = 2500,
    units = "px", res = 300)
par(mar = rep(0.5, 4))
set.seed(10); shuf_idx2 <- sample(1:nrow(umap_morph))
plot(x = umap_morph[shuf_idx2,1], 
     y = umap_morph[shuf_idx2,2],
     pch = 16,
     col = cluster_cols_morph[as.character(df_morph$celltype)][shuf_idx2],
     xaxt = "n",
     yaxt = "n",
     bty = "n",
     xlim = c(-.5,1.75),
     ylim = c(-1,1),
     xlab = "",
     ylab = "")

# put morphology
set.seed(10); shuf_idx <- sample(1:nrow(umap_gex))

points(x = umap_gex[shuf_idx,1], 
       y = umap_gex[shuf_idx,2],
       pch = 16,
       col = cluster_cols_gex[as.character(df_gex$celltype)][shuf_idx])
graphics.off()






