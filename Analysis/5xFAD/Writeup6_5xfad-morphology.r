rm(list=ls())

df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/5xFAD_CAJAL-only_umap.csv",
               row.names = 1)
plot_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup6/"

col_palette <- c(
  "0" = "#FF2F92",
  "1" = "#00E1FF",
  "2" = "#8C564B"
)


png(paste0(plot_folder, "Writeup6_CAJAL_umap.png"),
    height = 800, width = 1200, res = 300, units = "px")
plot(x = df[,"umap_1"],
     y = df[,"umap_2"],
     pch = 17,
     col = col_palette[as.character(df[,"Cluster"])],
     cex = 1,
     main = "0: Pink, 1: Cyan, 2: Brown")

graphics.off()

png(paste0(plot_folder, "Writeup6_CAJAL_umap_cleaned.png"),
    height = 800, width = 1200, res = 300, units = "px")
par(mar = rep(0.5,4))
plot(x = df[,"umap_1"],
     y = df[,"umap_2"],
     pch = 17,
     col = "black",
     xaxt = "n",
     yaxt = "n",
     bty = "n",
     cex = 2)

points(x = df[,"umap_1"],
       y = df[,"umap_2"],
       pch = 17,
       col = "white",
       xaxt = "n",
       yaxt = "n",
       bty = "n",
       cex = 1.5)

points(x = df[,"umap_1"],
       y = df[,"umap_2"],
       pch = 17,
       col = col_palette[as.character(df[,"Cluster"])],
       xaxt = "n",
       yaxt = "n",
       bty = "n",
       cex = 1)

graphics.off()

#############################

# do simple exploration
df_extended <- sapply(rownames(df), function(x){
  tmp <- strsplit(x, split = "_")
  tmp[[1]][1:5]
})
df_extended <- as.data.frame(t(df_extended))
colnames(df_extended) <- c("Region", "Model", "Time", "Sex", "Replicate")

df$cell_id <- rownames(df)
df_extended$cell_id <- rownames(df_extended)

df <- merge(x = df, 
            y = df_extended,
            by = "cell_id")
rownames(df) <- df$cell_id

table(df$Cluster, df$Time)
table(df$Cluster, df$Sex)

zz <- table(df$Cluster, df$Region); round(100*diag(1/rowSums(zz)) %*% zz)
zz <- table(df$Cluster, df$Time); round(100*diag(1/rowSums(zz)) %*% zz)
zz <- table(df$Cluster, df$Sex); round(100*diag(1/rowSums(zz)) %*% zz)

zz <- table(df$Cluster, df$Region); round(100*diag(1/rowSums(zz)) %*% zz)
zz <- table(df$Cluster, df$Time); round(100*diag(1/rowSums(zz)) %*% zz)
zz <- table(df$Cluster, df$Sex); round(100*diag(1/rowSums(zz)) %*% zz)
