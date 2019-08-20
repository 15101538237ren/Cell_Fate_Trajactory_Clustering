# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, re, math, time, glob, pickle
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import fastcluster as fc
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import directed_hausdorff
from numba import jit
#%matplotlib inline

# Some visualization stuff, not so important
sns.set()
plt.rcParams['figure.figsize'] = (12, 12)

# Utility Functions
color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                 'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])
Stages = [8, 10, 11, 12, 13, 14, 16, 18, 20, 22]
INDEX_RANGE = 1305

@jit(nopython=True)
def eucl_dist(x,y):
    """
    Usage
    -----
    L2-norm between point x and y
    Parameters
    ----------
    param x : numpy_array
    param y : numpy_array
    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    dist = np.linalg.norm(x-y)
    return dist
@jit(nopython=True)
def point_to_seg(p,s1,s2):
    """
    Usage
    -----
    Point to segment distance between point p and segment delimited by s1 and s2
    Parameters
    ----------
    param p : 1x2 numpy_array
    param s1 : 1x2 numpy_array
    param s2 : 1x2 numpy_array
    Returns
    -------
    dpl: float
         Point to segment distance between p and s
    """
    px = p[0]
    py = p[1]
    p1x = s1[0]
    p1y = s1[1]
    p2x = s2[0]
    p2y = s2[1]
    if p1x==p2x and p1y==p2y:
        dpl=eucl_dist(p,s1)
    else:
        segl= eucl_dist(s1,s2)
        u1 = (((px - p1x) * (p2x - p1x)) + ((py - p1y) * (p2y - p1y)))
        u = u1 / (segl * segl)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = eucl_dist(p,s1)
            iy = eucl_dist(p, s2)
            if ix > iy:
                dpl = iy
            else:
                dpl = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = p1x + u * (p2x - p1x)
            iy = p1y + u * (p2y - p1y)
            dpl = eucl_dist(p, np.array([ix, iy]))

    return dpl

def point_to_trajectory (p, t):
    """
    Usage
    -----
    Point-to-trajectory distance between point p and the trajectory t.
    The Point to trajectory distance is the minimum of point-to-segment distance between p and all segment s of t
    Parameters
    ----------
    param p: 1x2 numpy_array
    param t : len(t)x2 numpy_array
    Returns
    -------
    dpt : float,
          Point-to-trajectory distance between p and trajectory t
    """
    dpt=min(map(lambda s1,s2 : point_to_seg(p,s1,s2), t[:-1],t[1:] ))
    return dpt

def e_spd (t1, t2):
    """
    Usage
    -----
    The spd-distance of trajectory t2 from trajectory t1
    The spd-distance is the sum of the all the point-to-trajectory distance of points of t1 from trajectory t2
    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array
    Returns
    -------
    spd : float
           spd-distance of trajectory t2 from trajectory t1
    """
    spd=sum(map(lambda p : point_to_trajectory(p,t2),t1))/len(t1)
    return spd

def e_sspd (t1, t2):
    """
    Usage
    -----
    The sspd-distance between trajectories t1 and t2.
    The sspd-distance is the mean of the spd-distance between of t1 from t2 and the spd-distance of t2 from t1.
    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array
    Returns
    -------
    sspd : float
            sspd-distance of trajectory t2 from trajectory t1
    """
    sspd= e_spd(t1,t2)
    return sspd

# ####################
# Pairwise Distance #
# ####################

def pdist(traj_list, metric="sspd", type_d="euclidean", implementation="auto"):
    """
    Usage
    -----
    Pairwise distances between trajectory in traj_list.
    metrics available are :
    1. 'sspd'
        Computes the distances using the Symmetrized Segment Path distance.
    2. 'dtw'
        Computes the distances using the Dynamic Path Warping distance.
    3. 'lcss'
        Computes the distances using the Longuest Common SubSequence distance
    4. 'hausdorf'
        Computes the distances using the Hausdorff distance.
    5. 'frechet'
        Computes the distances using the Frechet distance.
    6. 'discret_frechet'
        Computes the distances using the Discrete Frechet distance.
    7. 'sowd_grid'
        Computes the distances using the Symmetrized One Way Distance.
    8. 'erp'
        Computes the distances using the Edit Distance with real Penalty.
    9. 'edr'
        Computes the distances using the Edit Distance on Real sequence.
    type_d available are "euclidean" or "geographical". Some distance can be computing according to geographical space
    instead of euclidean. If so, traj_0 and traj_1 have to be 2-dimensional. First column is longitude, second one
    is latitude.
    If the distance traj_0 and traj_1 are 2-dimensional, the cython implementation is used else the python one is used.
    unless "python" implementation is specified
    'sowd_grid' computes distance between trajectory in grid representation. If the coordinate
    are geographical, this conversion can be made according to the geohash encoding. If so, the geohash 'precision'
    is needed.
    'edr' and 'lcss' require 'eps' parameter. These distance assume that two locations are similar, or not, according
    to a given threshold, eps.
    'erp' require g parameter. This distance require a gap parameter. Which must have same dimension that the
    trajectory.
    Parameters
    ----------
    param traj_list:       a list of nT numpy array trajectory
    param metric :         string, distance used
    param type_d :         string, distance type_d used (geographical or euclidean)
    param implementation : string, implementation used (python, cython, auto)
    param converted :      boolean, specified if the data are converted in cell format (sowd_grid
                           metric)
    param precision :      int, precision of geohash (sowd_grid )
    param eps :            float, threshold distance (edr and lcss)
    param g :              numpy arrays, gaps (erp distance)
    Returns
    -------
    M : a nT x nT numpy array. Where the i,j entry is the distance between traj_list[i] and traj_list[j]
    """
    nb_traj = len(traj_list)
    print("Computing " + type_d + " distance " + metric + " with implementation " + implementation + " for %d trajectories" % nb_traj)
    M = np.zeros(sum(range(nb_traj)))
    im=0
    for i in range(nb_traj):
        traj_list_i = traj_list[i]
        for j in range(i + 1, nb_traj):
            traj_list_j = traj_list[j]
            M[im] = e_sspd(traj_list_i, traj_list_j)
            im += 1
    return M

def convert_data_into_np_array(data_save_pick_fp, load=True, number_of_pcs = 2):
    traj_list = []
    pair_indexs = range(1, INDEX_RANGE + 4)
    if load == False:
        STAGE_DATA_DIR = os.path.join(os.path.abspath(os.curdir), 'DATA', 'stage_data')
        Prob2d_DATA_DIR = os.path.join(os.path.abspath(os.curdir), 'DATA', 'stage_prob2d_data')
        data_dict = {}
        for dir_name in os.listdir(STAGE_DATA_DIR):
            if os.path.isdir(os.path.join(STAGE_DATA_DIR, dir_name)):
                dir_index = int(dir_name.split("_")[0])
                if dir_index <= INDEX_RANGE:
                    data_fp = os.path.join(STAGE_DATA_DIR, dir_name, "stage_metrics.csv")
                    df = pd.read_csv(data_fp, sep=",", header=0).values
                    data_dict[dir_index] = {}
                    data_dict[dir_index]['pair_name'] = '_'.join(re.sub(r'[\[|\]|\\|\'|\s]', '', df[0, 4]).split(','))
                    pcs = df[:, 7: 7 + number_of_pcs].astype(float)
                    data_dict[dir_index]['pcs'] = pcs

                    data_dict[dir_index]['prob2d'] = []
                    stage_prob2d_gene_pair_dir = os.path.join(Prob2d_DATA_DIR, dir_name)
                    for sid, stage in enumerate(Stages):
                        wild_card_path = os.path.join(stage_prob2d_gene_pair_dir, "Stage_%d_*.mat" % stage)
                        for filename in glob.glob(wild_card_path):
                            data_fp = os.path.join(stage_prob2d_gene_pair_dir, filename)
                            prob2d_arr = sio.loadmat(data_fp)
                            data_dict[dir_index]['prob2d'].append(prob2d_arr['prob2d'])

        pairs_name = ["sox2-t", "gata5-pax8", "lhx1-pax8"]
        for pid, pair in enumerate(pairs_name):
            base_dir = os.path.join(os.path.abspath(os.curdir), "DATA", "target_gene_pairs")
            input_fp = os.path.join(base_dir, pair + ".csv")
            pcs = pd.read_csv(input_fp, sep=",", header=0).values
            d_id = INDEX_RANGE + pid + 1
            data_dict[d_id] ={}
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
        traj = data_dict[key]['pcs']
        traj_list.append(traj)
        # plt.plot(traj[:, 0], traj[:, 1])
    # plt.show()
    return [data_dict, traj_list]

def colorline(ax, x, y, z=None, linestyle = 'solid', cmap='gist_rainbow', norm=plt.Normalize(0.0, 1.0),
        linewidth=1, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

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

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


# 3 - Distance matrix

def hausdorff(u, v):
    d = np.linalg.norm(u - v) * 3 #max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0]) +
    return d

def calc_distance_matrix(distance_fp, traj_lst, metric_name):
    ts = time.time()
    #p_dist = pdist(traj_list, metric=metric_name)
#     nb_dist = len(p_dist)
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


def plot_cluster(N_CLUSTER, traj_lst, cluster_lst, fig_fp):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''

    N_COL = 5
    N_ROW = int(math.ceil(float(N_CLUSTER) / N_COL))
    c_arr = np.array([(time_point + 1.) / 10. for time_point in range(10)])
    traj_lst = np.array(traj_lst)
    cluster_lst = np.array(cluster_lst)
    fig, axs = plt.subplots(N_ROW, N_COL, figsize=(N_COL * EACH_SUB_FIG_SIZE, N_ROW * EACH_SUB_FIG_SIZE))
    for index, (traj, cluster) in enumerate(zip(traj_lst, cluster_lst)):
        row = cluster // N_COL
        col = cluster % N_COL
        if N_ROW > 1:
            ax = axs[row][col]
        else:
            ax = axs[cluster]
        data_index = index + 1
        if data_index > INDEX_RANGE:
            colorline(ax, traj[:, 0], traj[:, 1], c_arr, linestyle='dashed')
            ax.text(0, 0, data_dict[data_index]['pair_name'], color='black')
        else:

            colorline(ax, traj[:, 0], traj[:, 1], c_arr)
        ax.set_xlim(-20, 45)
        ax.set_ylim(-20, 40)
        ax.set_title("Cluster %d" % (cluster + 1))
    plt.savefig(fig_fp, dpi=200)
    plt.show()

def mkdirs(dir_paths):
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def trajectory_segmentation(traj_lst, degree_threshold = 5):
    for traj_index, traj in enumerate(traj_lst):

        hold_index_lst = []
        previous_azimuth = 1000

        for point_index, point in enumerate(traj[:-1]):
            next_point = traj[point_index + 1]
            diff_vector = next_point - point
            azimuth = (math.degrees(math.atan2(*diff_vector)) + 360) % 360

            if abs(azimuth - previous_azimuth) > degree_threshold:
                hold_index_lst.append(point_index)
                previous_azimuth = azimuth
        hold_index_lst.append(traj.shape[0] - 1)  # Last point of trajectory is always added

        traj_lst[traj_index] = traj[hold_index_lst, :]
    return traj_lst



def plot_heatmap_serie_of_each_cluster(data_dict, N_CLUSTER, cluster_lst, fig_dir):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''
    data_dict = data_dict[0]
    N_COL = 10
    Max_NROW = 5
    TICKS = range(0, 21, 5)
    N_SUBFIG_PER_FIG = Max_NROW * N_COL
    cluster_lst = np.array(cluster_lst)
    for cluster in range(N_CLUSTER):
        gene_pair_indexs = np.where(cluster_lst == cluster)[0]
        n_gene_pairs_in_cluster = len(gene_pair_indexs)
        NFIG = int(math.ceil(float(n_gene_pairs_in_cluster) / Max_NROW))
        sub_fig_dir = os.path.join(fig_dir, "Cluster_%d" % (cluster + 1))
        mkdirs([sub_fig_dir])
        for i in range(NFIG):
            if NFIG > 1:
                fig_fp = os.path.join(sub_fig_dir, "Cluster_%d_%d.png" % (cluster + 1, i))
            else:
                fig_fp = os.path.join(sub_fig_dir, "Cluster_%d.png" % (cluster + 1))
            base_index = i * N_SUBFIG_PER_FIG
            N_remaining_files = n_gene_pairs_in_cluster * N_COL - base_index
            N_ROW = int(math.ceil(float(N_remaining_files) / N_COL)) if N_remaining_files <= N_SUBFIG_PER_FIG else Max_NROW
            plt.clf()
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


if __name__ == "__main__":
    data_save_pick_fp = os.path.join(os.path.abspath(os.curdir), "DATA", "All_Data.pkl")
    [data_dict, traj_list] = convert_data_into_np_array(data_save_pick_fp, load=False)
    metric = "directed_hausdorff"
    distance_fp = "DATA/pair_distance_by_%s_d2.npy" % metric
    calc_distance_matrix(distance_fp, traj_list, metric)
    EACH_SUB_FIG_SIZE = 4
    N_CLUSTER = 10
    fig_dir = 'figure_outputs/Traj_clusters'
    mkdirs([fig_dir])
    fig_fp = os.path.join(fig_dir, "cluster_by_%s.png" % metric)
    p_dist = np.load(distance_fp)
    Z = fc.linkage(p_dist, method="ward")
    labels = sch.fcluster(Z, t=2500, criterion="distance") - 1#2500
    N_CLUSTER = len(np.unique(labels))
    plot_cluster(N_CLUSTER, traj_list, labels, fig_fp)
    # plot_heatmap_serie_of_each_cluster([data_dict], N_CLUSTER, labels, fig_dir)
