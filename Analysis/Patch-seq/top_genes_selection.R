data<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/output/writeup1/laplacian_scores(5000).csv")
head(data)
quantile(data$laplacian_q_values)
number<-data$laplacian_q_values<=0.05364249  #25% quantile
selected<-(1:dim(data)[1])[number]
selected_genes<-data$X[selected]

exon<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/exon_data_norm.csv")
selected_genes1 <- intersect(selected_genes, colnames(exon))
selected_exon<-exon[,selected_genes1]

write.csv(selected_exon, file = "/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/selected_exon.csv", row.names = FALSE)


n1<-quantile(data$laplacian_q_values,0.05)
number<-data$laplacian_q_values<=n1  #5% quantile
selected<-(1:dim(data)[1])[number]
selected_genes<-data$X[selected]
exon<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/exon_data_norm.csv")
selected_genes1 <- intersect(selected_genes, colnames(exon))
selected_exon<-exon[,selected_genes1]

write.csv(selected_exon, file = "/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/selected_exon_5.csv", row.names = FALSE)


n1<-quantile(data$laplacian_q_values,0.1)
number<-data$laplacian_q_values<=n1  #10% quantile
selected<-(1:dim(data)[1])[number]
selected_genes<-data$X[selected]
exon<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/exon_data_norm.csv")
selected_genes1 <- intersect(selected_genes, colnames(exon))
selected_exon<-exon[,selected_genes1]

write.csv(selected_exon, file = "/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/selected_exon_10.csv", row.names = FALSE)



n1<-quantile(data$laplacian_q_values,0.5)
number<-data$laplacian_q_values<=n1  #50% quantile
selected<-(1:dim(data)[1])[number]
selected_genes<-data$X[selected]
exon<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/exon_data_norm.csv")
selected_genes1 <- intersect(selected_genes, colnames(exon))
selected_exon<-exon[,selected_genes1]

write.csv(selected_exon, file = "/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/selected_exon_50.csv", row.names = FALSE)


head(data)
quantile(data$laplacian_q_values)
number<-data$laplacian_q_values>=0.47800612  #25% quantile reversed
selected<-(1:dim(data)[1])[number]
selected_genes<-data$X[selected]
exon<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/exon_data_norm.csv")
selected_genes1 <- intersect(selected_genes, colnames(exon))
selected_exon<-exon[,selected_genes1]

write.csv(selected_exon, file = "/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/selected_exon_25_reversed.csv", row.names = FALSE)


head(data)
n1<-quantile(data$laplacian_q_values,0.95)
number<-data$laplacian_q_values>=n1  #5% quantile reversed
selected<-(1:dim(data)[1])[number]
selected_genes<-data$X[selected]
exon<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/exon_data_norm.csv")
selected_genes1 <- intersect(selected_genes, colnames(exon))
selected_exon<-exon[,selected_genes1]

write.csv(selected_exon, file = "/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/selected_exon_5_reversed.csv", row.names = FALSE)



head(data)
n1<-quantile(data$laplacian_q_values,0.9)
number<-data$laplacian_q_values>=n1  #10% quantile reversed
selected<-(1:dim(data)[1])[number]
selected_genes<-data$X[selected]
exon<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/exon_data_norm.csv")
selected_genes1 <- intersect(selected_genes, colnames(exon))
selected_exon<-exon[,selected_genes1]

write.csv(selected_exon, file = "/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/selected_exon_10_reversed.csv", row.names = FALSE)




head(data)
n1<-quantile(data$laplacian_q_values,0.5)
number<-data$laplacian_q_values>=n1  #50% quantile reversed
selected<-(1:dim(data)[1])[number]
selected_genes<-data$X[selected]
exon<-read.csv("/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/exon_data_norm.csv")
selected_genes1 <- intersect(selected_genes, colnames(exon))
selected_exon<-exon[,selected_genes1]

write.csv(selected_exon, file = "/Users/apple/Desktop/KLin_Group/Project_2024/data/Was2CODE_analysis_turbo/selected_exon_50_reversed.csv", row.names = FALSE)
