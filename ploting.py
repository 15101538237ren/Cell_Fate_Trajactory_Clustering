# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, math
import matplotlib.pyplot as plt

FIGURE_DIR= os.path.join(os.path.abspath(os.curdir), 'Figures')

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}% ({v:d})'.format(p=pct,v=val) if pct > 0 else ''
    return my_autopct

def plot_lineage_percentage():
    INDEX_RANGE = 1380
    stage_data_dir_name = "stage_data_log_training"
    run_name = "%d_gene_pairs_%s" % (INDEX_RANGE, stage_data_dir_name)

    N_CLUSTER = 17
    N_COL = 5
    N_ROW = int(math.ceil(float(N_CLUSTER) / N_COL))
    EACH_SUB_FIG_SIZE = 5

    fig, axs = plt.subplots(N_ROW, N_COL, figsize=(N_COL * EACH_SUB_FIG_SIZE, N_ROW * EACH_SUB_FIG_SIZE))
    lineage_fp = os.path.join(os.path.abspath(os.curdir), "DATA", "gene_pair_lineage.csv")
    lineage_df = pd.read_csv(lineage_fp, sep=",", header=0).values
    lineages = np.unique(lineage_df[:, 1]) #[item for item in  if item.split("-")[0] not in ["S14", "S16", "S18"]]
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * clid / len(lineages)) for clid in range(len(lineages))]
    for cid in range(N_COL * N_ROW):
        row = cid // N_COL
        col = cid % N_COL
        if N_ROW > 1:
            ax = axs[row][col]
        else:
            ax = axs[cid]
        if cid < N_CLUSTER:
            gene_pair_name_fp = os.path.join(FIGURE_DIR, run_name, "cluster_%d" % (cid + 1),
                                             "gene_pair_names_of_cluster_%d.csv" % (cid + 1))
            gp_indexs_in_cluster = pd.read_csv(gene_pair_name_fp, sep=",", header=None).values[:, 0]
            lineages_in_cluster = lineage_df[np.isin(lineage_df[:, 0], gp_indexs_in_cluster), 1]
            lineage_names, counts = np.unique(lineages_in_cluster, return_counts=True)
            #percentage = counts/np.sum(counts)
            lineage_count_dict = {lineage : 0 for lineage in lineages}
            for lid, ln in enumerate(lineage_names):
                lineage_count_dict[ln] = counts[lid]
            count_arr = [lineage_count_dict[lineage] for lineage in lineages]
            ax.pie(count_arr, autopct=make_autopct(counts), startangle=90, colors= colors)
            if cid == 0:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.3, box.height])
                ax.legend(lineages, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title("cluster %d" % (cid + 1))
        else:
            ax.axis('off')
    fig_fp = os.path.join(FIGURE_DIR, run_name, "lineage_of_each_cluster.png")
    plt.savefig(fig_fp, transparent=True, bbox_inches='tight', pad_inches=0.1, format='png')

if __name__ == "__main__":
    plot_lineage_percentage()