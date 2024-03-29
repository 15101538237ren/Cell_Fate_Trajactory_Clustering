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
for (i in seq(from=40, to=n_cluster))
{
  gene_pair_fp = paste(cluster_gene_pair_dir, paste0("cluster_", i), paste0("gene_pair_names_of_cluster_", i, ".csv") , sep = "/")
  gene_pair_df = as.data.frame(read.csv(gene_pair_fp, header = F))
  colnames(gene_pair_df)<-c("gene_pair_index", "gene_name1", "gene_name2")
  lineages = as.character(lineage_df[lineage_df$Index %in% gene_pair_df$gene_pair_index, ]$Root_node)
  gnames_in_gene_pairs = c(as.character(gene_pair_df$gene_name1),as.character(gene_pair_df$gene_name2))
  
  selected_gene_df = gene_df[which(gene_df$gene_symbol %in% gnames_in_gene_pairs), ]
  tgn = as.data.frame(table(as.character(selected_gene_df$gene_symbol)))
  tgn = tgn[order(tgn$Freq, decreasing = T),]
  tgn = cbind(tgn, tgn$Freq / length(gnames_in_gene_pairs))
  colnames(tgn)<-c("gene_symbol", "Freq", "Ratio")
  rownames(tgn) <- tgn$gene_symbol
  topn_tgn = tgn#[1:TOP_N, ]
  top_gene_df = selected_gene_df[which(selected_gene_df$gene_symbol %in% tgn$gene_symbol) , ]
  gene_symb = as.character(topn_tgn$gene_symbol)
  selected_go_df = go_df[which(go_df$gene_symbol %in% gene_symb), ]
  if (nrow(selected_go_df) == 0)
  {
    selected_go_df <- go_df%>% dplyr::filter(gene_symbol %like% gene_symb)
    #selected_go_df = selected_go_df[1:TOP_N, ]
    selected_go_df$gene_symbol = gene_symb
  }
  go_ids = as.character(selected_go_df$GO_IDs)
  go_description_vec = c()
  for (j in 1:length(go_ids))
  {
    go_description = c()
    go_list= strsplit(go_ids[j], ',')[[1]]
    for (k in 1:length(go_list))
    {
      description = as.character(goterms[which(goterms$GO_IDs == go_list[k]), ]$Description)
      go_description = c(go_description, description)
    }
    go_description_vec =c(go_description_vec, str_c(go_description, collapse = ";"))
  }
  
  go_description_df = as.data.frame(gene_symb, go_description_vec)
  go_description_df = cbind(go_description_df, rownames(go_description_df))
  colnames(go_description_df) = c("gene_symbol", "GO_Description")
  rownames(go_description_df) = go_description_df$gene_symbol
  merged_data = merge(merge(top_gene_df, topn_tgn, by="gene_symbol"), merge(selected_go_df, go_description_df, by="gene_symbol"), by="gene_symbol")
  symb_to_query = as.character(merged_data$gene_symbol)
  entrez_ids_df = entrez_gene_ids[which(entrez_gene_ids$alias_symbol %in% symb_to_query), ]
  if (nrow(entrez_ids_df) == 0)
  {
    entrez_ids_df <- entrez_gene_ids[pmatch(symb_to_query, entrez_gene_ids$alias_symbol),]
    entrez_ids_df$alias_symbol = symb_to_query
  }
  
  entrez_ids_df =entrez_ids_df[1:TOP_N, ]
  fuction_df <- as.data.frame(UniProt.ws::select(up, keys = as.character(entrez_ids_df$gene_id) ,columns = c("FUNCTION"), keytype = "ENTREZ_GENE"))
  if (nrow(na.omit(fuction_df)) ==0)
  {
    fuction_df[is.na(fuction_df)] <- ''
    fuction_df <- aggregate(FUNCTION ~ ENTREZ_GENE, data = fuction_df, paste, collapse = ";")
  } else {
    #fuction_df = na.omit(fuction_df)#[1:TOP_N, ]
    fuction_df[is.na(fuction_df)] <- ''
  }
  
  merged_final_data = merge(merged_data, merge(fuction_df, entrez_ids_df, by.x="ENTREZ_GENE", by.y="gene_id"), by.x="gene_symbol", by.y="alias_symbol")
  
  cluster_ids = as.data.frame(rep(i, nrow(merged_final_data)))
  colnames(cluster_ids) = c("cluster_id")
  merged_final_data = cbind(cluster_ids, merged_final_data)
  selected_out_df = merged_final_data %>% select(cluster_id, gene_symbol, gene_name, gene_function, Freq, FUNCTION, GO_Description)
  selected_out_df = selected_out_df %>% filter(!duplicated(gene_symbol, FUNCTION))
  if (nrow(all_gene_description_to_export) == 0) {
    all_gene_description_to_export = selected_out_df
  } else {
    all_gene_description_to_export = rbind(all_gene_description_to_export, selected_out_df)
  }
}
write.csv(all_gene_description_to_export, out_csv_fp)