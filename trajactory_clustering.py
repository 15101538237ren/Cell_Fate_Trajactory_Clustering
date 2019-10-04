# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, re, math, time, glob, pickle, warnings
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import fastcluster as fc
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# from scipy.spatial.distance import directed_hausdorff
warnings.filterwarnings('ignore')
from matplotlib import collections as mcoll
from Util import mkdirs
#%matplotlib inline

# Utility Functions
color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                 'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])
DATA_DIR=os.path.join(os.path.abspath(os.curdir), 'DATA')
PICKLE_DATA= os.path.join(DATA_DIR, "pickle_data")
NPY_DATA= os.path.join(DATA_DIR, "npy_data")
GENE_PAIR_NAME_DATA = os.path.join(DATA_DIR, "gene_pair_names")
FIGURE_DIR= os.path.join(os.path.abspath(os.curdir), 'Figures')
mkdirs([PICKLE_DATA, NPY_DATA, FIGURE_DIR, GENE_PAIR_NAME_DATA])
Stages = [8, 10, 11, 12, 13, 14, 16, 18, 20, 22]
EACH_SUB_FIG_SIZE = 5
FIGURE_FORMAT = "png"

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(ax, x, y, z=None, linestyle = 'solid', cmap='gist_rainbow', norm=plt.Normalize(0.0, 1.0), linewidth=1, alpha=1.0):
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha, linestyle=linestyle)

    ax.add_collection(lc)
    return lc

def convert_data_into_np_array(STAGE_DATA_DIR_NAME, INDEX_RANGE, data_save_pick_fp, load=True, number_of_pcs=2, OFF_SET = 0, include_sox_and_t=True, Filter=True):
    traj_list = []
    gene_pair_names = []
    sox_t_count = 1 if include_sox_and_t else 0
    pair_indexs = []
    if load == False:
        STAGE_DATA_DIR = os.path.join(DATA_DIR, STAGE_DATA_DIR_NAME)
        data_dict = {}
        for dir_name in os.listdir(STAGE_DATA_DIR):
            data_dir = os.path.join(STAGE_DATA_DIR, dir_name)
            if os.path.isdir(data_dir):
                dir_index = int(dir_name.split("_")[0])
                if dir_index <= INDEX_RANGE:
                    data_fp = os.path.join(data_dir, "stage_metrics.csv")
                    df = pd.read_csv(data_fp, sep=",", header=0).values
                    pcs = df[:, 7 + OFF_SET: 7 + OFF_SET + number_of_pcs].astype(float)
                    if (Filter == True and np.sum(pcs[:, 0] > 20) > (len(Stages) / 2.)) or Filter==False:
                        pair_indexs.append(dir_index)
                        data_dict[dir_index] = {}
                        data_dict[dir_index]['pair_name'] = '_'.join(re.sub(r'[\[|\]|\\|\'|\s]', '', df[0, 4+OFF_SET]).split(','))
                        data_dict[dir_index]['pcs'] = pcs

                        data_dict[dir_index]['prob2d'] = []
                        for sid, stage in enumerate(Stages):
                            wild_card_path = os.path.join(data_dir, "Stage_%d_*.mat" % stage)
                            for filename in glob.glob(wild_card_path):
                                data_fp = os.path.join(data_dir, filename)
                                prob2d_arr = sio.loadmat(data_fp)
                                data_dict[dir_index]['prob2d'].append(prob2d_arr['prob2d'])

        pairs_name = ["sox2-t"] if include_sox_and_t else []
        for pid, pair in enumerate(pairs_name):
            base_dir = os.path.join(DATA_DIR, "target_gene_pairs")
            input_fp = os.path.join(base_dir, pair + ".csv")
            pcs = pd.read_csv(input_fp, sep=",", header=0).values
            d_id = INDEX_RANGE + pid + 1
            data_dict[d_id] = {}
            # the elements of each entry in data dict
            data_dict[d_id]['pair_name'] = pair
            data_dict[d_id]['pcs'] = pcs
            data_dict[d_id]['prob2d'] = []

            stage_prob2d_gene_pair_dir = os.path.join(base_dir, pair)
            for sid, stage in enumerate(Stages):
                wild_card_path = os.path.join(stage_prob2d_gene_pair_dir, "Stage_%d_*.mat" % stage)
                for filename in glob.glob(wild_card_path):
                    data_fp = os.path.join(stage_prob2d_gene_pair_dir, filename)
                    prob2d_arr = sio.loadmat(data_fp)
                    data_dict[d_id]['prob2d'].append(prob2d_arr['prob2D'])
        with open(data_save_pick_fp, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(data_save_pick_fp, 'rb') as handle:
            data_dict = pickle.load(handle)
    for key in pair_indexs:
        traj_list.append(data_dict[key]['pcs'])
        gene_pair_names.append(data_dict[key]['pair_name'])
        # plt.plot(traj[:, 0], traj[:, 1])
    # plt.show()
    return [pair_indexs, data_dict, traj_list, gene_pair_names]


def hausdorff(u, v):
    # 3 - Distance matrix
    d = np.linalg.norm(u - v)  # + max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d


def calc_distance_matrix(distance_fp, traj_lst):
    ts = time.time()
    traj_count = len(traj_lst)
    D = np.zeros((traj_count, traj_count))

    # This may take a while
    for i in range(traj_count):
        for j in range(i + 1, traj_count):
            distance = hausdorff(traj_lst[i], traj_lst[j])
            D[i, j] = distance
            D[j, i] = distance
    te = time.time()
    print("%d Distances computed in %d seconds" % (traj_count, te - ts))
    np.save(distance_fp, D)


def plot_cluster(traj_lst, cluster_lst, run_name, fig_format="png", color_palette=None, log_transformed=True, cmap="gist_rainbow"):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''
    if log_transformed:
        X_MAX_LIM = 65
        Y_MAX_LIM = 40
        X_MIN_LIM = -17.14
        Y_MIN_LIM = -22.23
    else:
        X_MAX_LIM = 0.6
        Y_MAX_LIM = 0.15
        X_MIN_LIM = -0.2
        Y_MIN_LIM = -0.1
    N_CLUSTER = len(np.unique(cluster_lst))
    cluster_mapping = {cluster : cid for cid, cluster in enumerate(np.unique(cluster_lst))}
    N_COL = 4
    N_ROW = int(math.ceil(float(N_CLUSTER) / N_COL))
    c_arr = np.array([(time_point + 1.) / 10. for time_point in range(10)])
    traj_lst = np.array(traj_lst)
    cluster_lst = np.array(cluster_lst)
    fig, axs = plt.subplots(N_ROW, N_COL, figsize=(N_COL * EACH_SUB_FIG_SIZE, N_ROW * EACH_SUB_FIG_SIZE))
    for index, (traj, cluster) in enumerate(zip(traj_lst, cluster_lst)):
        row = cluster_mapping[cluster] // N_COL
        col = cluster_mapping[cluster] % N_COL
        if N_ROW > 1:
            ax = axs[row][col]
        else:
            ax = axs[col]
        colorline(ax, traj[:, 0], traj[:, 1], c_arr, cmap=cmap)
        ax.set_xlim(X_MIN_LIM, X_MAX_LIM)
        ax.set_ylim(Y_MIN_LIM, Y_MAX_LIM)
        if row==N_ROW - 1:
            ax.set_xlabel("PCA Component 1")
        if col == 0:
            ax.set_ylabel("PCA Component 2")
        if color_palette:
            ax.set_title("cluster %d" % (cluster_mapping[cluster] + 1), backgroundcolor=color_palette[cluster_mapping[cluster]])
        else:
            ax.set_title("cluster %d" % (cluster_mapping[cluster] + 1))

    fig_dir = os.path.join(FIGURE_DIR, run_name)
    mkdirs([fig_dir])
    fig_fp = os.path.join(fig_dir, "trajactory_clusters_%d.%s" % (N_CLUSTER, fig_format))
    plt.savefig(fig_fp, transparent=True, bbox_inches='tight', pad_inches=0.1, format='png')
    #plt.show()
    return [cluster_mapping[cluster] for cluster in cluster_lst]

def plot_heatmap_serie_of_each_cluster(data_dict, N_CLUSTER, cluster_lst, passed_pair_indexs, run_name, fig_format="png",
                                       TARGET_CLUSTER_IDs=None):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''
    print(N_CLUSTER)
    data_dict = data_dict[0]
    N_COL = 10
    Max_NROW = 5
    TICKS = range(0, 21, 5)
    N_SUBFIG_PER_FIG = Max_NROW * N_COL
    cluster_lst = np.array(cluster_lst)
    fig_dir = os.path.join(FIGURE_DIR, run_name)
    mkdirs([fig_dir])
    if TARGET_CLUSTER_IDs:
        cluster_ids = TARGET_CLUSTER_IDs
    else:
        cluster_ids = range(N_CLUSTER)
    for cluster in cluster_ids:
        gene_pair_indexs = passed_pair_indexs[cluster_lst == cluster]
        n_gene_pairs_in_cluster = len(gene_pair_indexs)
        NFIG = int(math.ceil(float(n_gene_pairs_in_cluster) / Max_NROW))
        sub_fig_dir = os.path.join(fig_dir, "cluster_%d" % (cluster + 1))
        mkdirs([sub_fig_dir])
        for i in range(NFIG):
            if NFIG > 1:
                fig_fp = os.path.join(sub_fig_dir, "cluster_%d_%d.%s" % (cluster + 1, i, fig_format))
            else:
                fig_fp = os.path.join(sub_fig_dir, "cluster_%d.%s" % (cluster + 1, fig_format))
            base_index = i * N_SUBFIG_PER_FIG
            N_remaining_files = n_gene_pairs_in_cluster * N_COL - base_index
            N_ROW = int(
                math.ceil(float(N_remaining_files) / N_COL)) if N_remaining_files <= N_SUBFIG_PER_FIG else Max_NROW

            fig, axs = plt.subplots(N_ROW, N_COL, figsize=(N_COL * EACH_SUB_FIG_SIZE, N_ROW * EACH_SUB_FIG_SIZE))
            SUB_FIG_RANGE = N_SUBFIG_PER_FIG if N_remaining_files > N_SUBFIG_PER_FIG else N_remaining_files
            plt.set_cmap('viridis_r')
            for j in range(SUB_FIG_RANGE):
                row = j // N_COL
                col = j % N_COL
                if N_ROW == 1:
                    ax = axs[col]
                else:
                    ax = axs[row][col]
                gene_pair_id = gene_pair_indexs[i * Max_NROW + row] + 1
                prob2d_array = data_dict[gene_pair_id]['prob2d']  # shape 21* 21
                prob2d = prob2d_array[col]
                q_potential = -np.log(np.abs(prob2d))
                cax = ax.pcolormesh(q_potential, vmin=3, vmax=14)
                ax.set_yticks(TICKS)
                ax.set_xticks(TICKS)
                if row == 0:
                    ax.set_title("Stage %d" % Stages[col])
                if col == 0:
                    ax.set_ylabel(data_dict[gene_pair_id]['pair_name'])
                if j == 0:
                    fig.colorbar(cax, ax=ax)
            plt.savefig(fig_fp, dpi=200)
            print("cluster %d" % (cluster + 1))

def merge_cluster_by_mannual_assignment(cluster_to_merge_list, labels):
    N_CLUSTER = len(cluster_to_merge_list)
    merged_labels = np.array([label for label in labels])
    for cid, cluster_ids in enumerate(cluster_to_merge_list):
        for cluster_id in cluster_ids:
            merged_labels[merged_labels == cluster_id - 1] = cid
    return merged_labels

def plot_hierarchical_cluster(df, linkage, color_palette, distance_threshold, label_arr, link_cols, run_name, fig_format="png", label_position=-400):

    n_cluster = len(np.unique(label_arr))
    row_colors = df.cluster_label.map(color_palette)
    fig_dir = os.path.join(FIGURE_DIR, run_name)
    mkdirs([fig_dir])
    #     cm = sns.clustermap(df, method="ward", col_cluster=True, col_colors=row_colors,  yticklabels=True, figsize=(35, 35))
    #     fig_fp1 = os.path.join(fig_dir, "hierarchical_cluster_%d.%s" % (n_cluster, fig_format))
    #     cm.savefig(fig_fp1, dpi=200)

    fig2 = plt.figure(2, figsize=(30, 30))
    R = dendrogram(linkage, no_labels=True, leaf_rotation=90, orientation="top",
                   leaf_font_size=8, distance_sort='ascending', color_threshold=distance_threshold,
                   above_threshold_color="black",
                   link_color_func=lambda x: link_cols[x])
    prev_sum = 0
    for lbl_id in range(n_cluster):
        label_tmp = label_arr[label_arr == lbl_id].shape[0] * 10
        coord = prev_sum + label_tmp * 0.3
        prev_sum += label_tmp
        plt.text(coord, label_position, "cluster %d" % (lbl_id + 1), rotation=45, backgroundcolor=color_palette[lbl_id])
    plt.ylim([label_position - 20, linkage[-1, -2]])
    fig_fp2 = os.path.join(fig_dir, "cluster_dendrogram_%d_with_leafs.%s" % (n_cluster, fig_format))
    fig2.savefig(fig_fp2, dpi=200)

def get_unique_gene_pairs_names_in_each_cluster(run_name, gene_pairs, cluster_labels_of_gene_pairs):
    output_dir = os.path.join(GENE_PAIR_NAME_DATA, run_name)
    #output_dir = os.path.join(FIGURE_DIR, run_name)
    mkdirs([output_dir])

    unique_gene_names = []
    cluster_labels_of_gene_pairs = np.array(cluster_labels_of_gene_pairs)
    gene_pairs = np.array(gene_pairs)
    n_cluster = len(np.unique(cluster_labels_of_gene_pairs))
    for cluster_id in range(n_cluster):
        gene_pairs_in_this_cluster = gene_pairs[cluster_labels_of_gene_pairs == cluster_id]
        all_gene_names_in_this_cluster = []
        for gene_pair_name in gene_pairs_in_this_cluster:
            for gene_name in gene_pair_name.split("_"):
                all_gene_names_in_this_cluster.append(gene_name)
        unique_gene_names_in_this_cluster = np.unique(all_gene_names_in_this_cluster)
        unique_gene_names.append(unique_gene_names_in_this_cluster)
        output_fp = os.path.join(output_dir, "%d.csv" % (cluster_id + 1))

        np.savetxt(output_fp, np.array(unique_gene_names_in_this_cluster)[:], delimiter='\n', fmt="%s")

def write_gene_pairs_names_in_each_cluster(run_name, gene_pair_indexs, gene_pair_names, cluster_labels_of_gene_pairs):
    output_dir = os.path.join(FIGURE_DIR, run_name)
    mkdirs([output_dir])
    cluster_labels_of_gene_pairs = np.array(cluster_labels_of_gene_pairs)
    gene_pairs = np.array(gene_pair_names)
    # cluster_mapping = {cid: cluster  for cid, cluster in enumerate(np.unique(cluster_labels_of_gene_pairs))}
    Ncluster = len(np.unique(cluster_labels_of_gene_pairs))
    for cluster_id in range(Ncluster):
        gene_pairs_in_this_cluster = gene_pairs[cluster_labels_of_gene_pairs == cluster_id]
        print(gene_pairs_in_this_cluster)
        gene_pairs_indexs_in_this_cluster = gene_pair_indexs[cluster_labels_of_gene_pairs == cluster_id]
        print(gene_pairs_indexs_in_this_cluster)
        out_arr = np.array([[gene_pairs_indexs_in_this_cluster[idx], gene_pair_name.split("_")[0],gene_pair_name.split("_")[1]] for idx, gene_pair_name in enumerate(gene_pairs_in_this_cluster)])
        outdir = os.path.join(output_dir,  "cluster_%d" % (cluster_id + 1))
        mkdirs([outdir])
        output_fp = os.path.join(outdir, "gene_pair_names_of_cluster_%d.csv"  % (cluster_id + 1))
        np.savetxt(output_fp, out_arr[:, :], delimiter='\n', fmt="%s,%s,%s")

def filter_cluster(traj_list, labels, pair_indexs, gene_pair_names, threshold = 20):
    NCLUSTER = len(np.unique(labels))
    traj_list = np.array(traj_list)
    labels = np.array(labels)
    pair_indexs = np.array(pair_indexs)
    gene_pair_names = np.array(gene_pair_names)

    passed_traj_list = np.array([])
    passed_labels = np.array([])
    passed_pair_indexs = np.array([])
    passed_gene_pair_names = np.array([])

    for cid in np.arange(NCLUSTER):
        traj_index = np.nonzero(labels == cid)
        trajs = traj_list[traj_index]
        n_passed = np.sum(np.array([1 for traj in trajs if np.any(traj > threshold)]))
        if n_passed > trajs.shape[0]/2.:
            passed_traj_list = trajs if passed_traj_list.shape[0] == 0 else np.concatenate((passed_traj_list, trajs), axis=0)
            passed_labels = labels[traj_index] if passed_labels.shape[0] == 0 else np.concatenate((passed_labels, labels[traj_index]), axis=0)
            passed_pair_indexs = pair_indexs[traj_index] if passed_pair_indexs.shape[0] == 0 else np.concatenate((passed_pair_indexs, pair_indexs[traj_index]), axis=0)
            passed_gene_pair_names = gene_pair_names[traj_index] if passed_gene_pair_names.shape[0] == 0 else np.concatenate((passed_gene_pair_names, gene_pair_names[traj_index]), axis=0)

    return [passed_traj_list, passed_labels, passed_pair_indexs, passed_gene_pair_names]


def fig8_plot_selected_cluster_trajectories(selected_clusters, cluster_labels, traj_lst,
                                            run_name, N_SAMPLE=4, fig_format='png', cmap="RdBu", color_palette=None):
    selected_clusters = [item - 1 for item in selected_clusters]
    N_CLUSTER = len(selected_clusters)
    plt.rc('axes', linewidth=2., labelsize=10., labelpad=6.)  # length for x tick and padding
    plt.rc('xtick.major', size=6, pad=3)  # length for x tick and padding
    plt.rc('ytick.major', size=6, pad=3)  # length for y tick and padding
    plt.rc('lines', mew=5, lw=4)  # line 'marker edge width' and thickness
    plt.rc('ytick', labelsize=8)  # ytick label size
    plt.rc('xtick', labelsize=8)  # xtick label
    plt.rc('figure', dpi=300)  # Sets rendering resolution to 300
    plt.rc('lines', markersize=3.1)  # Sets marker size for scatter to reasonable size

    c_arr = np.array([(time_point + 1.) / 10. for time_point in range(10)])
    for cid, cluster in enumerate(selected_clusters):
        fig = plt.figure(1000 + cluster)
        trajectories = np.array(traj_lst)[np.array(cluster_labels) == cluster]
        # n_sample = N_SAMPLEx
        # sampled_indexs = np.random.choice(np.arange(trajectories.shape[0]), min(trajectories.shape[0], n_sample),
        #                                   replace=False)
        # sampled_trajectories = trajectories[sampled_indexs]

        for traj in trajectories:
            colorline(plt.gca(), traj[:, 0], traj[:, 1], c_arr, cmap=cmap)
        plt.xlim(-17.14, 65)
        plt.ylim(-22.23, 40)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        #         plt.title(titles_of_each_cluster[cid])
        fig_dir = os.path.join(FIGURE_DIR, run_name)
        mkdirs([fig_dir])
        fig_fp = os.path.join(fig_dir, "selected_cluster_%d.%s" % ((cluster + 1), fig_format))
        plt.savefig(fig_fp, transparent=True, bbox_inches='tight', pad_inches=0.1, format='png')

if __name__ == "__main__":
    INDEX_RANGE = 1380
    stage_data_dir_name = "stage_data_log_training"
    log_transformed = True
    OFF_SET = 1
    include_sox_and_t = True

    run_name = "%d_gene_pairs_%s" % (INDEX_RANGE, stage_data_dir_name)
    pickle_fp = os.path.join(PICKLE_DATA, "%s.pkl" % run_name)
    [pair_indexs, data_dict, traj_list, gene_pair_names] = convert_data_into_np_array(stage_data_dir_name, INDEX_RANGE, pickle_fp,
                                                                         load=False, OFF_SET=OFF_SET,
                                                                         include_sox_and_t=include_sox_and_t, Filter=False)

    metric = "directed_hausdorff_plus_pair_wise_euclidean"  # pair_wise_euclidean_distance
    distance_fp = os.path.join(NPY_DATA, "%s_%s.npy" % (run_name, metric))
    calc_distance_matrix(distance_fp, traj_list)
    cm = plt.get_cmap('gist_rainbow')
    p_dist = np.load(distance_fp)
    Z = fc.linkage(p_dist, method="ward")
    distance_threshold = 800 if log_transformed else 10
    labels = fcluster(Z, t=distance_threshold, criterion="distance") - 1
    [passed_traj_list, passed_labels, passed_pair_indexs, passed_gene_pair_names] = filter_cluster(traj_list, labels, pair_indexs, gene_pair_names)
    CLUSTER_PLOT_CMAP = "gist_rainbow"
    passed_labels = plot_cluster(passed_traj_list, passed_labels, run_name, FIGURE_FORMAT, color_palette=None,
                 log_transformed=log_transformed, cmap=CLUSTER_PLOT_CMAP)

    # calc_distance_matrix(distance_fp, passed_traj_list)
    # p_dist = np.load(distance_fp)
    # Z = fc.linkage(p_dist, method="ward")
    # passed_labels = fcluster(Z, t=distance_threshold, criterion="distance") - 1
    #
    # label_position = -180
    # N_LABEL_CATEGORY = len(np.unique(passed_labels))
    # df = pd.DataFrame(data=p_dist, index=passed_gene_pair_names, columns=passed_gene_pair_names)
    # df['cluster_label'] = passed_labels
    # color_palette = dict(
    #     zip(df.cluster_label.unique(), [cm(1. * i / N_LABEL_CATEGORY) for i in range(N_LABEL_CATEGORY)]))
    # # plot_cluster(passed_traj_list, passed_labels, run_name, FIGURE_FORMAT, color_palette,
    # #              log_transformed=log_transformed, cmap=CLUSTER_PLOT_CMAP)
    # #
    write_gene_pairs_names_in_each_cluster(run_name, passed_pair_indexs, passed_gene_pair_names, passed_labels)
    # D_leaf_colors = {gid: rgb2hex(color_palette[passed_labels[gid]]) for gid, gpn in enumerate(passed_gene_pair_names)}  #
    # link_cols = {}
    # dflt_col = "#808080"
    # for i, i12 in enumerate(Z[:, :2].astype(int)):
    #     c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x] for x in i12)
    #     link_cols[i + 1 + len(Z)] = c1 if c1 == c2 else dflt_col
    #
    # plot_hierarchical_cluster(df, Z, color_palette, distance_threshold, passed_labels, link_cols, run_name, fig_format="png",
    #                           label_position=label_position)
    # TARGET_CLUSTER_IDs = None
    # plot_heatmap_serie_of_each_cluster([data_dict], N_LABEL_CATEGORY, passed_labels, passed_pair_indexs, run_name, FIGURE_FORMAT,
    #                                    TARGET_CLUSTER_IDs)
    # selected_clusters = [8, 11, 19, 22]
    # N_SAMPLE = 15
    # cmap = CLUSTER_PLOT_CMAP
    # fig8_plot_selected_cluster_trajectories(selected_clusters, passed_labels, passed_traj_list,
    #                                         run_name=run_name, N_SAMPLE=N_SAMPLE, cmap=cmap, fig_format='png')

