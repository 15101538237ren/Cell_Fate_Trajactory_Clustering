import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import seaborn as sns
import pandas as pd

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

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def mkdirs(dir_paths):
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

if __name__ == "__main__":
    # Data set
    url = 'mtcars.csv'
    df = pd.read_csv(url)
    df = df.set_index('model')

    # Prepare a vector of color mapped to the 'cyl' column
    my_palette = dict(zip(df.cyl.unique(), ["orange", "yellow", "brown"]))
    row_colors = df.cyl.map(my_palette)

    # plot
    sns.clustermap(df, metric="correlation", method="single", cmap="Blues", standard_scale=1, row_colors=row_colors)
