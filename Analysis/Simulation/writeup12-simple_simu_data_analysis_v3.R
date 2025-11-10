rm(list=ls())
library(Seurat)
library(scCustomize)
library(ggplot2)
library(ggrepel)
library(dplyr)
library(tidyr)

gex_metadata <- read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Simulation/v3/cluster_assignments.csv")
rownames(gex_metadata) <-paste0("neuron_", 1:1000)
#gex_metadata<-gex_metadata[,-1]


count_mat <- read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Simulation/v3/gene_expression.csv")
rownames(count_mat) <- paste0("neuron_", 1:1000)
colnames(count_mat) <- gsub("_", "-", colnames(count_mat))

head(rownames(gex_metadata))
head(colnames(t(count_mat)))
sum(rownames(gex_metadata)==colnames(t(count_mat)))


seurat_obj <- Seurat::CreateSeuratObject(counts = t(count_mat),
                                         meta.data = gex_metadata)
seurat_obj <- Seurat::NormalizeData(seurat_obj)
#seurat_obj <- Seurat::FindVariableFeatures(seurat_obj, 
#                                           selection.method = "vst", 
#                                           nfeatures = 2000)
all.genes <- rownames(seurat_obj)
seurat_obj <- ScaleData(seurat_obj)


#seurat_obj <- RunPCA(seurat_obj, features = all.genes)
#seurat_obj <- RunUMAP(seurat_obj, dims = 1:2, reduction = "pca")

expr_mat <- t(GetAssayData(seurat_obj, assay = "RNA", layer = "scale.data"))
seurat_obj[["no_pca"]] <- CreateDimReducObject(
  embeddings = expr_mat,
  key = "PC_",
  assay = DefaultAssay(seurat_obj)
)
seurat_obj <- RunUMAP(seurat_obj, dims = 1:3, reduction = "no_pca")
DimPlot(seurat_obj, reduction = "umap", group.by = "cluster")

seurat_obj <- FindNeighbors(seurat_obj, dims = 1:2) 
seurat_obj <- FindClusters(seurat_obj, resolution = 0.1)
DimPlot(seurat_obj, reduction = "umap", group.by = "seurat_clusters")


##########GW Dist#########
rm(list=ls())
library(Seurat)
library(scCustomize)
library(ggplot2)
library(ggrepel)
library(dplyr)
library(tidyr)

gex_metadata <- read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Simulation/v3/cluster_assignments.csv")
rownames(gex_metadata) <-paste0("neuron_", 1:1000)
count_mat <- read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Simulation/v3/gw_dist.csv")
count_mat<-count_mat[,-1]
rownames(count_mat)<-paste0("neuron_", 1:1000)

seurat_obj <- Seurat::CreateSeuratObject(counts = t(count_mat),
                                         meta.data = gex_metadata)
seurat_obj <- Seurat::NormalizeData(seurat_obj)
seurat_obj <- Seurat::FindVariableFeatures(seurat_obj, 
                                           selection.method = "vst", 
                                           nfeatures = 200)
seurat_obj <- Seurat::ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj, 
                     features = VariableFeatures(seurat_obj), 
                     approx = FALSE)
seurat_obj <- Seurat::RunUMAP(seurat_obj, 
                              dims = 1:30)

# umap
Seurat::DimPlot(seurat_obj,
                group.by = "cluster")

seurat_obj <- Seurat::FindNeighbors(seurat_obj, dims = 1:10)
seurat_obj <- Seurat::FindClusters(seurat_obj, resolution = 0.01)
Seurat::DimPlot(seurat_obj,
                group.by = "seurat_clusters")

###########find example point#########
library(ggplot2)
library(ggrepel)
library(dplyr)
library(tidyr)


umap_coords <- Seurat::Embeddings(seurat_obj, "umap")
metadata <- seurat_obj@meta.data
plot_data <- data.frame(
  UMAP_1 = umap_coords[,1],
  UMAP_2 = umap_coords[,2],
  cluster = metadata$seurat_clusters,
  cell_id = rownames(metadata)
)

set.seed(123)
selected_cells <- plot_data %>%
  group_by(cluster) %>%
  summarise(
    centroid_cell = {
      cluster_center_x <- mean(UMAP_1)
      cluster_center_y <- mean(UMAP_2)
      distances <- sqrt((UMAP_1 - cluster_center_x)^2 + (UMAP_2 - cluster_center_y)^2)
      cell_id[which.min(distances)]
    },
    random_cell = {
      remaining_cells <- cell_id[cell_id != cell_id[which.min(sqrt((UMAP_1 - mean(UMAP_1))^2 + (UMAP_2 - mean(UMAP_2))^2))]]
      sample(remaining_cells, 1)
    },
    .groups = 'drop'
  )


selected_cells_long <- selected_cells %>%
  pivot_longer(cols = c(centroid_cell, random_cell), 
               names_to = "selection_type", 
               values_to = "cell_id") %>%
  mutate(selection_type = ifelse(selection_type == "centroid_cell", "Centroid", "Random"))


annotation_data <- selected_cells_long %>%
  left_join(plot_data, by = "cell_id") %>%
  mutate(
    label = paste0("C", cluster.y, "_",  
                   ifelse(selection_type == "Centroid", "Cen", "Ran")),
    full_label = paste0(label, "\n", cell_id)
  ) %>%
  rename(cluster = cluster.y) %>%  
  select(-cluster.x)  


p <- ggplot(plot_data, aes(x = UMAP_1, y = UMAP_2, color = cluster)) +
  geom_point(size = 1, alpha = 0.7) +

  geom_point(data = annotation_data, 
             aes(x = UMAP_1, y = UMAP_2, shape = selection_type), 
             color = "black", size = 1, stroke = 1, fill = "white") +
  scale_shape_manual(values = c("Centroid" = 21, "Random" = 22)) +

  geom_text_repel(data = annotation_data,
                  aes(x = UMAP_1, y = UMAP_2, label = full_label),
                  color = "black",
                  size = 3,
                  box.padding = 0.5,
                  point.padding = 0.3,
                  segment.color = "gray50",
                  segment.size = 0.5,
                  max.overlaps = Inf,
                  force = 3,
                  min.segment.length = 0) +
  theme_classic() +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    plot.title = element_text(size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5)
  ) +
  labs(
    title = "UMAP Plot with Selected Cells per Cluster",
    subtitle = "Circle = Centroid, Square = Random",
    x = "UMAP 1",
    y = "UMAP 2",
    color = "Cluster",
    shape = "Selection Type"
  ) +
  guides(
    color = guide_legend(override.aes = list(size = 3)),
    shape = guide_legend(override.aes = list(size = 3))
  )

print(p)


cat("Selected cells for each cluster:\n")
selected_summary <- annotation_data %>%
  select(cluster, selection_type, cell_id, label) %>%
  arrange(cluster, selection_type)
print(selected_summary)


write.csv(
  selected_summary$cell_id,
  "/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Simulation/v2/simu_simplfied_1/selected_cells_per_cluster.csv",
  row.names = FALSE
)


com_mat<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Simulation/v4/gw_dist.csv")
com_mat<-com_mat[,-1]
rownames(com_mat) <-paste0("neuron_", 1:3000)
colSums(com_mat)
var(colSums(com_mat)[1:1800])


write.csv(
  com_mat,
  "/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Simulation/v3/coupling_probability_matrix.csv",
)

celltype<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Simulation/v3/cluster_label.csv")
rownames(celltype) <-paste0("neuron_", 1:1000)
write.csv(
  celltype,
  "/Users/apple/Desktop/KLin_Group/Project_2024/data/Morpho_data/dataset/Simulation/v3/celltype.csv",
)


install.packages("pheatmap")
library(pheatmap)
pheatmap(com_mat, 
         scale = "none",        
         cluster_rows = FALSE,  
         cluster_cols = FALSE,
         show_rownames = FALSE, 
         show_colnames = FALSE)

