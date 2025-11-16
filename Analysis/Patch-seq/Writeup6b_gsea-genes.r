rm(list=ls())

library(clusterProfiler)
library(org.Mm.eg.db)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggrepel)

plot_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup6b/"

df <- read.csv("/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/csv/kevin/Writeup6_turbo-csvs/patchseq_GeoAdvAE_gene_importance_ranked.csv",
               row.names = 1)

teststat_vec <- df[,"mean_attribution"] 
names(teststat_vec) <- rownames(df)
teststat_vec <- teststat_vec[!is.na(teststat_vec)]
teststat_vec <- sort(teststat_vec, decreasing = TRUE) # Sort in decreasing order

# Run GSEA
set.seed(10)
gse <- clusterProfiler::gseGO(
  teststat_vec,
  ont = "BP", # Biological Process ontology
  keyType = "SYMBOL",
  OrgDb = "org.Mm.eg.db",
  pvalueCutoff = 1,       # p-value threshold for pathways
  minGSSize = 10,         # minimum gene set size
  maxGSSize = 500,         # maximum gene set size
  scoreType = ifelse(all(teststat_vec > 0), "pos", "std")
)

gse_df <- as.data.frame(gse)
gse_df2 <- gse_df[gse_df$pvalue <= 0.05,]
head(gse_df2$Description)
head(gse_df2$p.adjust)

gse_df2[c(2,6,9,10,12,14,17,19,22,23,24),]

gse_df2[c("GO:0097485", "GO:0007411"),]
gse_df2[c("GO:0051056"),]

############################

# --- 1) Define which GO terms map to each color group ---
go_red  <- c("GO:0097485", "GO:0007411")   # neuron projection guidance, axon guidance
go_blue <- c("GO:0051056")                 # regulation of small GTPase mediated signal transduction

# --- 2) Extract core-enrichment genes for each group from gse_df2 ---
extract_core_genes <- function(df, ids) {
  df %>%
    filter(ID %in% ids) %>%
    select(core_enrichment) %>%
    separate_rows(core_enrichment, sep = "/") %>%
    transmute(gene = str_trim(core_enrichment)) %>%
    filter(nzchar(gene)) %>%
    distinct() %>%
    pull(gene)
}

genes_red  <- extract_core_genes(gse_df2, go_red)
genes_blue <- extract_core_genes(gse_df2, go_blue)

# --- 3) Build plotting frame from your named vector, sorted decreasing ---
df <- tibble(
  gene  = names(teststat_vec),
  value = as.numeric(teststat_vec)
) %>%
  arrange(desc(value)) %>%
  mutate(
    order       = row_number(),
    color_group = case_when(
      gene %in% genes_red  ~ "red",
      gene %in% genes_blue ~ "blue",
      TRUE                 ~ "black"
    ),
    color_group = factor(color_group, levels = c("black","blue","red")) # layer order
  )

# (optional) sanity messages
missing_red  <- setdiff(genes_red,  df$gene)
missing_blue <- setdiff(genes_blue, df$gene)
if (length(missing_red))  message("Red genes not in teststat_vec: ",  paste(missing_red,  collapse=", "))
if (length(missing_blue)) message("Blue genes not in teststat_vec: ", paste(missing_blue, collapse=", "))

# --- 4) Plot: black first, then blue, then red (so blue/red sit on top) ---
plot1 <- ggplot(df, aes(order, value)) +
  geom_point(data = ~filter(.x, color_group == "black"),
             color = "black", size = 1.2, alpha = 0.9) +
  geom_point(data = ~filter(.x, color_group == "blue"),
             color = "blue",  size = 1.8) +
  geom_point(data = ~filter(.x, color_group == "red"),
             color = "red",   size = 1.8) +
  geom_text_repel(
    data = ~filter(.x, color_group != "black"),
    aes(label = gene, color = color_group),
    size = 3, show.legend = FALSE, max.overlaps = Inf, min.segment.length = 0
  ) +
  scale_color_manual(values = c(black="black", blue="blue", red="red")) +
  labs(x = "Rank (decreasing value)", y = "teststat",
       title = "Test statistics ranked by value",
       subtitle = "red: axon/projection guidance • blue: small GTPase signaling • black: other") +
  theme_minimal(base_size = 12)

ggplot2::ggsave(plot1,
                file = paste0(plot_folder, "Writeup6b_gsea.png"),
                height = 5, width = 8)

###########

# --- 4) Plot: black first, then blue, then red (so blue/red sit on top) ---
plot1 <- ggplot(df, aes(order, value)) +
  geom_point(data = ~filter(.x, color_group == "black"),
             color = "black", size = 1.2, alpha = 0.9) +
  geom_point(data = ~filter(.x, color_group == "blue"),
             color = "blue",  size = 1.8) +
  geom_point(data = ~filter(.x, color_group == "red"),
             color = "red",   size = 1.8) +
  geom_text_repel(
    data = ~filter(.x, color_group != "black"),
    aes(label = gene, color = color_group),
    size = 3.5, 
    show.legend = FALSE, 
    max.overlaps = Inf, 
    min.segment.length = 0,
    box.padding = 0.5,
    segment.alpha = 0.5
  ) +
  scale_color_manual(values = c(black="black", blue="blue", red="red")) +
  theme_minimal(base_size = 12)

ggplot2::ggsave(plot1,
                file = paste0(plot_folder, "Writeup6b_gsea_cleaned.png"),
                height = 2.5, width = 5)

