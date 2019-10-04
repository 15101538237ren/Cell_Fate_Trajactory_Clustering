library(dplyr)
library(stringr)

n_cluster <- 17
TOP_N = 5
cluster_gene_pair_dir = "~/PycharmProjects/Cell_Fate_Trajactory_Clustering/Figures/1380_gene_pairs_stage_data_log_training"
out_csv_fp = "~/PycharmProjects/Cell_Fate_Trajactory_Clustering/DATA/Gene_names_and_frequency_of_each_cluster.csv"
all_gene_description_to_export = data.frame()
for (i in seq(from=1, to=n_cluster))
{
  gene_pair_fp = paste(cluster_gene_pair_dir, paste0("cluster_", i), paste0("gene_pair_names_of_cluster_", i, ".csv") , sep = "/")
  gene_pair_df = as.data.frame(read.csv(gene_pair_fp, header = F))
  colnames(gene_pair_df)<-c("gene_pair_index", "gene_name1", "gene_name2")
  gnames_in_gene_pairs = c(as.character(gene_pair_df$gene_name1),as.character(gene_pair_df$gene_name2))
  tgn = as.data.frame(table(gnames_in_gene_pairs))
  tgn = tgn[order(tgn$Freq, decreasing = T),]
  tgn = cbind(tgn, rep(length(gnames_in_gene_pairs), nrow(tgn)))
  tgn = cbind(tgn, tgn$Freq / length(gnames_in_gene_pairs))
  colnames(tgn)<-c("GeneSymbol", "Freq", "Tot_Gene", "Ratio")
  rownames(tgn) <- tgn$GeneSymbol
  
  cluster_ids = as.data.frame(rep(i, nrow(tgn)))
  colnames(cluster_ids) = c("cluster_id")
  merged_final_data = cbind(cluster_ids, tgn)
  selected_out_df = merged_final_data %>% select(cluster_id, GeneSymbol, Freq, Tot_Gene, Ratio)
  selected_out_df = selected_out_df %>% filter(!duplicated(GeneSymbol))
  selected_out_df$Ratio <- format(selected_out_df$Ratio, digits = 2)
  if (nrow(all_gene_description_to_export) == 0) {
    all_gene_description_to_export = selected_out_df
  } else {
    all_gene_description_to_export = rbind(all_gene_description_to_export, selected_out_df)
  }
}
write.csv(all_gene_description_to_export, out_csv_fp)