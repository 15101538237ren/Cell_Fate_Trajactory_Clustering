library(biomaRt)
library(rtracklayer)
library(UniProt.ws)
library(GO.db)
library(dplyr)
library(stringr)
library(org.Xl.eg.db)
up <- UniProt.ws(taxId=8355)
goterms <- as.data.frame(Term(GOTERM))
goterms <- cbind(rownames(goterms), goterms)
entrez_gene_ids <- as.data.frame(org.Xl.egALIAS2EG)
colnames(goterms)<-c("GO_IDs", "Description")

gene_fp<-"../DATA/GenePageGeneralInfo_ManuallyCurated.txt"
gene_df <- as.data.frame(read.delim(gene_fp, header = F, sep = "\t"))
gene_df <- gene_df[c(2,3,4,1,5,6)]
colnames(gene_df)<-c("gene_symbol", "gene_name", "gene_function","Xenbase_page_id", "gene_synonyms", "JGI_ID")
gene_df <- gene_df[!duplicated(gene_df$gene_symbol), ]

lineage_fp = "../DATA/gene_pair_lineage.csv"
lineage_df <- as.data.frame(read.csv(lineage_fp, header = T))

go_fp = "../DATA/GeneGoTerms.txt"
go_df <- as.data.frame(read.delim(go_fp, header = F, sep = "\t"))
go_df <- go_df[c(3,4,1,2)]
colnames(go_df)<-c("gene_symbol", "GO_IDs", "Xenbase_page_id", "Xenbase_id")
go_df <- go_df[!duplicated(go_df$gene_symbol), ]
rownames(go_df)<-go_df$gene_symbol
n_cluster <- 49
TOP_N = 5
cluster_gene_pair_dir = "../Figures/1380_gene_pairs_stage_data_log_training"
out_csv_fp = "../DATA/Descriptions_of_top_genes_in_each_cluster_with_freq.csv"
all_gene_description_to_export = data.frame()
for (i in seq(from=1, to=n_cluster))
{
  gene_pair_fp = paste(cluster_gene_pair_dir, paste0("cluster_", i), paste0("gene_pair_names_of_cluster_", i, ".csv") , sep = "/")
  gene_pair_df = as.data.frame(read.csv(gene_pair_fp, header = F))
  colnames(gene_pair_df)<-c("gene_pair_index", "gene_name1", "gene_name2")
  lineages = as.character(lineage_df[lineage_df$Index %in% gene_pair_df$gene_pair_index, ]$Root_node)
  gnames_in_gene_pairs = c(as.character(gene_pair_df$gene_name1),as.character(gene_pair_df$gene_name2))
  uniq_gene_names = unique(gnames_in_gene_pairs)
  selected_gene_df = data.frame()
  for (gsb in uniq_gene_names)
  {
    gsb_df <- gene_df%>% dplyr::filter(gene_symbol %in% c(gsb, paste0(gsb, ".L", sep=""), paste0(gsb, ".S", sep="")))
    if (nrow(selected_gene_df) == 0) {
      selected_gene_df = gsb_df
    } else {
      selected_gene_df = rbind(selected_gene_df, gsb_df)
    }
  }
  if (nrow(selected_gene_df) > 0)
  {
    #selected_gene_df = gene_df[which(gene_df$gene_symbol %in% gnames_in_gene_pairs), ]
    tgn = as.data.frame(table(as.character(selected_gene_df$gene_symbol)))
    tgn = tgn[order(tgn$Freq, decreasing = T),]
    tgn = cbind(tgn, tgn$Freq / length(gnames_in_gene_pairs))
    colnames(tgn)<-c("gene_symbol", "Freq", "Ratio")
    rownames(tgn) <- tgn$gene_symbol
    max_len = min(nrow(tgn), TOP_N)
    selected_gene_df = selected_gene_df[1:max_len, ]
    topn_tgn = tgn[1:max_len, ]
    
    merged_final_data = merge(selected_gene_df, topn_tgn,  by="gene_symbol")
    
    cluster_ids = as.data.frame(rep(i, nrow(merged_final_data)))
    colnames(cluster_ids) = c("cluster_id")
    merged_final_data = cbind(cluster_ids, merged_final_data)
    selected_out_df = merged_final_data %>% select(cluster_id, gene_symbol, gene_name, gene_function, Freq)
    selected_out_df = selected_out_df %>% filter(!duplicated(gene_symbol))
    if (nrow(all_gene_description_to_export) == 0) {
      all_gene_description_to_export = selected_out_df
    } else {
      all_gene_description_to_export = rbind(all_gene_description_to_export, selected_out_df)
    }
  }
}
write.csv(all_gene_description_to_export, out_csv_fp)