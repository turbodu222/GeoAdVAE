library(ggplot2)
library(ggrepel)
library(dplyr)

# 1. Read data
merged_scores <- read.csv('/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Scala/laplacian_IG_scores_combined.csv', 
                          row.names = 1)

cat("Data dimensions:", dim(merged_scores), "\n")
cat("Column names:", colnames(merged_scores), "\n")

# 2. Standardize Laplacian and IG scores
laplacian_normalized <- scale(merged_scores$feature_laplacians)[,1]
ig_normalized <- scale(merged_scores$mean_attribution)[,1]

# 3. Create plotting dataframe
plot_df <- data.frame(
  gene = rownames(merged_scores),
  laplacian = laplacian_normalized,
  ig = ig_normalized,
  stringsAsFactors = FALSE
)

# 4. Identify genes in TOP-LEFT region only (laplacian < -1 AND ig > 1)
plot_df$highlight <- ifelse(plot_df$laplacian < -1 & plot_df$ig > 1, 
                            "highlight", 
                            "other")
plot_df$label <- ifelse(plot_df$highlight == "highlight", 
                        plot_df$gene, 
                        "")

cat("\nNumber of highlighted genes in top-left:", sum(plot_df$highlight == "highlight"), "\n")

# 5. Get tighter axis limits (reduce white space)
x_limit <- c(min(plot_df$laplacian) * 1.02, max(plot_df$laplacian) * 1.02)
y_limit <- c(min(plot_df$ig) * 1.02, max(plot_df$ig) * 1.02)

# 6. Create plot
p <- ggplot(plot_df, aes(x = laplacian, y = ig)) +
  # Reference lines at 0
  geom_hline(yintercept = 0, linewidth = 0.5, color = "gray") +
  geom_vline(xintercept = 0, linewidth = 0.5, color = "gray") +
  
  # VERY LIGHT purple rectangle for TOP-LEFT region only (x < -1, y > 1)
  geom_rect(aes(xmin = x_limit[1], xmax = -1, ymin = 1, ymax = y_limit[2]),
            fill = rgb(230, 210, 255, maxColorValue = 255),  # Much lighter purple
            alpha = 0.25,  # Slightly more visible than before
            inherit.aes = FALSE) +
  
  # ONLY cutoff lines for top-left box (x = -1 and y = 1)
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  geom_vline(xintercept = -1, linetype = "dashed", color = "black") +
  
  # Plot points
  geom_point(data = plot_df[plot_df$highlight == "other", ],
             color = "gray60", 
             alpha = 0.5,
             size = 1.5) +
  geom_point(data = plot_df[plot_df$highlight == "highlight", ],
             color = "red", 
             alpha = 0.8,
             size = 2) +
  
  # Add gene labels for highlighted genes
  geom_text_repel(data = plot_df[plot_df$highlight == "highlight", ],
                  aes(label = label),
                  color = "red", 
                  size = 3,
                  max.overlaps = 30,
                  show.legend = FALSE) +
  
  # Axis settings - tighter limits
  scale_x_continuous(limits = x_limit, expand = c(0.02, 0.02)) +
  scale_y_continuous(limits = y_limit, expand = c(0.02, 0.02)) +
  
  # Labels
  labs(
    x = "Normalized Laplacian Score (feature_laplacians)",
    y = "Normalized IG Score (mean_attribution)",
    title = paste0("Laplacian vs IG Scores (Normalized)\n",
                   "# of Features: ", nrow(plot_df),
                   ", # Highlighted (Top-Left): ", sum(plot_df$highlight == "highlight"))
  ) +
  
  theme_minimal() +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5),
    axis.title.x = element_text(size = 11),
    axis.title.y = element_text(size = 11),
    panel.grid.minor = element_blank()  # Remove minor grid lines
  )

# 7. Save plot
ggsave('/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Scala/laplacian_vs_IG_scatter_highlighted_topleft.png',
       plot = p,
       width = 10,
       height = 8,
       dpi = 300)

cat("\nPlot saved successfully!\n")

# 8. Print highlighted genes (top-left only)
cat("\nHighlighted genes in TOP-LEFT region (Laplacian < -1 AND IG > 1):\n")
highlighted_genes <- plot_df[plot_df$highlight == "highlight", ]
highlighted_genes <- highlighted_genes[order(-highlighted_genes$ig), ]
print(highlighted_genes[, c("gene", "laplacian", "ig")])

# Show plot
print(p)