rm(list=ls())

library(Seurat)
out_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/out/morpho_integration/kevin/Writeup5/"
data_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/data/GSE150358/"
plot_folder <- "/Users/kevinlin/Library/CloudStorage/Dropbox/Collaboration-and-People/turbo/git/morpho_integration/fig/kevin/Writeup5/"

count_mat <- read.csv(paste0(data_folder, "GSE150358_Microglia_UMI_counts.tsv"), sep = "\t")
metadata <- read.csv(paste0(data_folder, "GSE150358_Microglia_metadata.tsv"), sep = "\t")
count_mat <- Matrix::Matrix(as.matrix(count_mat), sparse = TRUE)

gene_vec <- c(
  "Tmem119","P2ry12","Ivns1abp","Selplg","Stab1","Cd164","Crybb1","Ank","Spp1","Axl",
  "Csf1","Cst7","Cd9","Cadm1","Apoe","B2m","Cstb","Tyrobp","Timp2","Fth1",
  "Top2a","Mki67","Pbk","Racgap1","Tpx2","Cenpe","Cdca3","Mcm5","Pcna","Tyms",
  "Fen1","Mcm2","Mcm4","Rrm1","Ifit2","Ifit3","Irf7","Oasl2","Ifit1","Ifi209","Ifi213"
)
gene_vec <- intersect(gene_vec, rownames(count_mat))

seurat_obj <- Seurat::CreateSeuratObject(counts = count_mat, meta.data = metadata)

# organize the metadata
for(j in 1:ncol(seurat_obj@meta.data)){
  vec <- seurat_obj@meta.data[,j]
  if(all(is.character(vec)) & length(unique(vec)) <= sqrt(length(vec))){
    seurat_obj@meta.data[,j] <- factor(vec)
  }
}

set.seed(10)
seurat_obj <- Seurat::NormalizeData(seurat_obj)
seurat_obj <- Seurat::FindVariableFeatures(seurat_obj, nfeatures = 2000)
variable_genes <- Seurat::VariableFeatures(seurat_obj)
variable_genes <- unique(c(variable_genes, gene_vec))
Seurat::VariableFeatures(seurat_obj) <- variable_genes

seurat_obj <- Seurat::ScaleData(seurat_obj,
                                vars.to.regress = c("percent.mt", "nCount_RNA", "Trem2"))
seurat_obj <- Seurat::RunPCA(seurat_obj, 
                             features = Seurat::VariableFeatures(seurat_obj),
                             verbose = FALSE)
set.seed(10)
seurat_obj <- Seurat::RunUMAP(seurat_obj, dims = 1:25)

Seurat::DimPlot(seurat_obj,
                group.by = "microgliaClusterID")

summary(seurat_obj@meta.data)

Seurat::DimPlot(seurat_obj,
                group.by = "Trem2_antibody")
Seurat::DimPlot(seurat_obj,
                group.by = "Trem2")

########################


plot1 <- Seurat::DotPlot(seurat_obj,
                         features = gene_vec,
                         group.by = "microgliaClusterID")
ggplot2::ggsave(plot1, filename = paste0(plot_folder, "Writeup5_wang_dotplot.png"),
                height = 5, width = 25)


plot1 <- Seurat::DoHeatmap(seurat_obj,
                           features = gene_vec,
                           group.by = "microgliaClusterID")
ggplot2::ggsave(plot1, filename = paste0(plot_folder, "Writeup5_wang_heatmap.png"),
                height = 10, width = 25)

save(seurat_obj, 
     file = paste0(out_folder, "wang_microglia_cleaned.RData"))
