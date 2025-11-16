rm(list=ls())

plot_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup6/"

gex_df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/5xFAD_GeoAdvAE_GEX_UMAP_coor_5XFAD.csv",
                   row.names = 1)
morph_df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/5xFAD_GeoAdvAE_Morpho_UMAP_coor_5XFAD.csv",
                     row.names = 1)


gex_palette <- c(
  "0"="#0B559F", "1"="#565AD6", "2"="#CFE8FF", "4"="#8FC2F0", "5"="#074ED3",
  "7"="#FFCF70", "3"="#F39C34", "8"="#D04E00",
  "10"="#A7E08B", "11"="#2E9F51", "9"="#338538",
  "6"="#DB72F5"
)

morph_palette <- c(
  "0" = "#FF2F92",
  "1" = "#00E1FF",
  "2" = "#8C564B"
)

set.seed(10)
shuff_idx <- sample(nrow(gex_df))


png(paste0(plot_folder, "Writeup6_umap-integration.png"),
    height = 800, width = 1933, res = 300, units = "px")
par(mar = rep(0.5,4))

plot(gex_df[shuff_idx,"umap_1"], 
     gex_df[shuff_idx,"umap_2"],
     pch = 16,
     col = gex_palette[as.character(gex_df[shuff_idx,"cluster_label"])],
     cex = 0.5,
     xaxt = "n",
     yaxt = "n",
     bty = "n")

points(morph_df[,"umap_1"],
       morph_df[,"umap_2"],
       pch = 17,
       col = "black",
       cex = 2)

points(morph_df[,"umap_1"],
       morph_df[,"umap_2"],
       pch = 17,
       col = "white",
       cex = 1.5)

points(morph_df[,"umap_1"],
       morph_df[,"umap_2"],
       pch = 17,
       col = morph_palette[as.character(morph_df[,"cluster_label"])],
       cex = 1)

graphics.off()