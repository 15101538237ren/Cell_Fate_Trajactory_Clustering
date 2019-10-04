library(dplyr)
library(stringr)

n_cluster <- 17
TOP_N = 5
cluster_gene_pair_dir = "~/PycharmProjects/Cell_Fate_Trajactory_Clustering/Figures/1380_gene_pairs_stage_data_log_training"
out_csv_fp = "~/PycharmProjects/Cell_Fate_Trajactory_Clustering/DATA/Gene_names_and_frequency_of_each_cluster.csv"
all_gene_description_to_export = data.frame()

lineage_fp = "~/PycharmProjects/Cell_Fate_Trajactory_Clustering/DATA/gene_pair_lineage.csv"
lineage_df <- as.data.frame(read.csv(lineage_fp, header = T))

for (i in seq(from=1, to=n_cluster))
{
  gene_pair_fp = paste(cluster_gene_pair_dir, paste0("cluster_", i), paste0("gene_pair_names_of_cluster_", i, ".csv") , sep = "/")
  gene_pair_df = as.data.frame(read.csv(gene_pair_fp, header = F))
  colnames(gene_pair_df)<-c("gene_pair_index", "gene_name1", "gene_name2")
  gene_pair_indexs_in_this_cluster = gene_pair_df$gene_pair_index
  lineages=lineage_df[lineage_df$Index %in% gene_pair_indexs_in_this_cluster, ]$Root_node
  lineage_percentage = table(lineages)/sum(table(lineages))
  pie(lineage_percentage, lineage_percentage, legend("topright", lineages), col = rainbow(length(lineages)))
}
write.csv(all_gene_description_to_export, out_csv_fp)