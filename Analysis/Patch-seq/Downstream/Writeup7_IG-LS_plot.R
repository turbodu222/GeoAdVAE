rm(list=ls())

library(ggplot2)
library(ggrepel)
library(dplyr)

# 1. Read data
merged_scores <- read.csv(
  '/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/turbo/scala/laplacian_IG_scores_combined.csv',
  row.names = 1
)

# 2. Standardize Laplacian and IG scores
laplacian_normalized <- merged_scores$feature_laplacians
ig_normalized <- merged_scores$mean_attribution

# 3. Create plotting dataframe
plot_df <- data.frame(
  gene = rownames(merged_scores),
  laplacian = laplacian_normalized,
  ig = ig_normalized,
  stringsAsFactors = FALSE
)

# 4. Identify highlighted genes
lap_cutoff <- 0.98
ig_cutoff <- 0.01

plot_df$highlight <- ifelse(
  plot_df$laplacian < lap_cutoff & plot_df$ig > ig_cutoff,
  "highlight",
  "other"
)

plot_df$label <- ifelse(
  plot_df$highlight == "highlight",
  plot_df$gene,
  ""
)

cat("\nNumber of highlighted genes in region:", sum(plot_df$highlight == "highlight"), "\n")

# 5. Axis limits
x_limit <- range(plot_df$laplacian, na.rm = TRUE)
y_limit <- range(plot_df$ig, na.rm = TRUE)

# 6. Correlation
cor_val <- cor(plot_df$laplacian, plot_df$ig, use = "complete.obs", method = "pearson")
cor_text <- paste0("Pearson correlation = ", round(cor_val, 3))

# 7. Create plot
p <- ggplot(plot_df, aes(x = laplacian, y = ig)) +
  # Background rectangle
  geom_rect(
    data = plot_df[1,],
    aes(xmin = -Inf, xmax = lap_cutoff, ymin = ig_cutoff, ymax = Inf),
    fill = rgb(230, 210, 255, maxColorValue = 255),
    alpha = 0.75,
    inherit.aes = FALSE
  ) +
  
  # Reference lines at means
  geom_hline(yintercept = mean(plot_df$ig, na.rm = TRUE), linewidth = 0.5, color = "gray") +
  geom_vline(xintercept = mean(plot_df$laplacian, na.rm = TRUE), linewidth = 0.5, color = "gray") +
  
  # Cutoff lines
  geom_hline(yintercept = ig_cutoff, linetype = "dashed", color = "black") +
  geom_vline(xintercept = lap_cutoff, linetype = "dashed", color = "black") +
  
  # Points
  geom_point(
    data = plot_df[plot_df$highlight == "other", ],
    color = "gray60",
    alpha = 0.5,
    size = 1.5
  ) +
  geom_point(
    data = plot_df[plot_df$highlight == "highlight", ],
    color = "red",
    alpha = 0.8,
    size = 2
  ) +
  
  # Correlation / regression line
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "steelblue",
    linewidth = 0.8
  ) +
  
  # Labels
  geom_text_repel(
    data = plot_df[plot_df$highlight == "highlight", ],
    aes(label = label),
    color = "red",
    size = 3,
    max.overlaps = 30,
    show.legend = FALSE
  ) +
  
  # Labels
  labs(
    x = "Laplacian score",
    y = "Integrated gradient",
    title = paste0(
      cor_text, "\n",
      "# of Features: ", nrow(plot_df),
      ", # Highlighted: ", sum(plot_df$highlight == "highlight")
    )
  ) +
  
  theme_minimal() +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5),
    axis.title.x = element_text(size = 11),
    axis.title.y = element_text(size = 11),
    panel.grid.minor = element_blank()
  )

# 8. Save plot
ggsave(
  '/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup7/Writeup7_laplacian_vs_IG_scatter.png',
  plot = p,
  width = 5,
  height = 4,
  dpi = 300
)

#####################

genes <- plot_df[plot_df$highlight == "highlight", "gene"]
write.table(genes, 
            file = "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/kevin/Writeup7/Writeup7_IG-LS_genes.csv",
            row.names = FALSE,
            col.names = FALSE,
            quote = FALSE)


####################

# https://www.informatics.jax.org/go/term/GO:0048812

path <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/data/pathways/GO_0048812_mouse.txt"

gene_df <- read.table(
  path, 
  sep = "\t",          # Assumes tab-separated
  header = TRUE,       # Use TRUE if the first line is column names
  quote = "",          # CRITICAL: Tells R to ignore all quotes/apostrophes
  row.names = NULL,    # CRITICAL: Prevents 'duplicate row.names' error
  stringsAsFactors = FALSE,
  fill = TRUE          # Helps if some rows have fewer columns than others
)

gene_vec <- unique(gene_df$MGI.Gene.Marker.ID)

plot_df$highlight <- ifelse(
  plot_df$gene %in% gene_vec,
  "highlight",
  "other"
)

plot_df$label <- ifelse(
  plot_df$highlight == "highlight",
  plot_df$gene,
  ""
)

cat("\nNumber of pathway genes found in data:", sum(plot_df$highlight == "highlight"), "\n")

# 7. Create plot
p <- ggplot(plot_df, aes(x = laplacian, y = ig)) +
  # Background rectangle
  geom_rect(
    data = plot_df[1,],
    aes(xmin = -Inf, xmax = lap_cutoff, ymin = ig_cutoff, ymax = Inf),
    fill = rgb(230, 210, 255, maxColorValue = 255),
    alpha = 0.75,
    inherit.aes = FALSE
  ) +
  
  # Reference lines at means
  geom_hline(yintercept = mean(plot_df$ig, na.rm = TRUE), linewidth = 0.5, color = "gray") +
  geom_vline(xintercept = mean(plot_df$laplacian, na.rm = TRUE), linewidth = 0.5, color = "gray") +
  
  # Cutoff lines
  geom_hline(yintercept = ig_cutoff, linetype = "dashed", color = "black") +
  geom_vline(xintercept = lap_cutoff, linetype = "dashed", color = "black") +
  
  # Points
  geom_point(
    data = plot_df[plot_df$highlight == "other", ],
    color = "gray60",
    alpha = 0.5,
    size = 1.5
  ) +
  geom_point(
    data = plot_df[plot_df$highlight == "highlight", ],
    color = "red",
    alpha = 0.8,
    size = 2
  ) +
  
  # Correlation / regression line
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "steelblue",
    linewidth = 0.8
  ) +
  
  # Labels
  geom_text_repel(
    data = plot_df[plot_df$highlight == "highlight", ],
    aes(label = label),
    color = "red",
    size = 3,
    max.overlaps = 30,
    show.legend = FALSE
  ) +
  
  # Labels
  labs(
    x = "Laplacian score",
    y = "Integrated gradient",
    title = paste0(
      cor_text, "\n",
      "# of Features: ", nrow(plot_df),
      ", # Highlighted: ", sum(plot_df$highlight == "highlight")
    )
  ) +
  
  theme_minimal() +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5),
    axis.title.x = element_text(size = 11),
    axis.title.y = element_text(size = 11),
    panel.grid.minor = element_blank()
  )


# 8. Save plot
ggsave(
  '/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup7/Writeup7_laplacian_vs_IG_scatter_highlighted.png',
  plot = p,
  width = 5,
  height = 4,
  dpi = 300
)
