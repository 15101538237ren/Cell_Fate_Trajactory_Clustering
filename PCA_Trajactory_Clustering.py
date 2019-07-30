# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, re, math, scipy.io, random
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN
from Util import kMedoids
import traj_dist.distance as tdist
import fastcluster as fc
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pickle
#%matplotlib inline

# Some visualization stuff, not so important
sns.set()
plt.rcParams['figure.figsize'] = (12, 12)

# Utility Functions
color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                 'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])

def convert_data_into_np_array(number_of_pcs = 2):
    STAGE_DATA_DIR = 'DATA/stage_data'
    data_dict = {}
    traj_list = []
    pair_indexs = range(1, 1306)
    for dir_name in os.listdir(STAGE_DATA_DIR):
        if os.path.isdir(os.path.join(STAGE_DATA_DIR, dir_name)):
            dir_index = int(dir_name.split("_")[0])
            data_fp = os.path.join(STAGE_DATA_DIR, dir_name, "stage_metrics.csv")
            df = pd.read_csv(data_fp, sep=",", header=0).values
            data_dict[dir_index] = {}
            data_dict[dir_index]['pair_name'] = ' vs '.join(re.sub(r'[\[|\]|\\|\'|\s]', '', df[0, 4]).split(','))
            pcs = df[:, 7: 7 + number_of_pcs]
            data_dict[dir_index]['pcs'] = pcs
    for key in pair_indexs:
        traj = data_dict[key]['pcs']
        traj_list.append(traj)
        plt.plot(traj[:, 0], traj[:, 1])
    plt.show()
    return [data_dict, traj_list]
def calc_distance_matrix(distance_fp, traj_list):
    ts = time.time()
    p_dist = tdist.pdist(traj_list, metric="sspd")
    nb_dist = len(p_dist)
    te = time.time()
    print("%d Distances computed in %d seconds" % (nb_dist, te - ts))
    np.save(distance_fp, p_dist)

if __name__ == "__main__":
    [data_dict, traj_list] = convert_data_into_np_array()
    distance_fp = 'DATA/pair_distance_matrix.npy'
    calc_distance_matrix(distance_fp, traj_list)
