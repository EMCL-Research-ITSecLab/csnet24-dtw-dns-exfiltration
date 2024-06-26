import matplotlib.pyplot as plt
import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch import cdist
from tslearn.neighbors import KNeighborsTimeSeries

from utils import HEICLOUD_DATA


def plot_neighbours(X, y, type: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    knn = KNeighborsTimeSeries(n_neighbors=len(X_train))
    knn.fit(X_train)

    q = X_test[1].reshape(1, X_test.shape[1], 1)
    ret = knn.kneighbors(q)
    nn = ret[1][0][0]  # The nearest neighbour
    fn = ret[1][0][-1]  # The farthest neighbor
    nn, fn

    plt.plot(q[0], linewidth=2, color="r")
    plt.plot(X_train[nn], linewidth=2, color="b")
    plt.title("The Nearest Neighbour")
    plt.savefig(f"figs/{type}_nearest_neighbour.pdf")
    plt.show()
    plt.clf()

    plt.plot(q[0], linewidth=2, color="r")
    plt.plot(X_train[fn], linewidth=2, color="g")
    plt.title("The Farthest Neighbour")
    plt.savefig(f"figs/{type}_farthest_neighbour.pdf")
    plt.show()
    plt.clf()


def plot_distribution(X, y, type: str):
    """Plots data distribution of NumPy Arrays

    Args:
        X (numpy.array): NumPy Array
    """
    ax = plt.axes()
    ax.xaxis.grid(which="both")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time")
    for idx, j in enumerate(X):
        if y[idx] == 1:
            c = "r"
        else:
            c = "b"

        plt.plot(j, linewidth=0.1, color=c)

    plt.minorticks_on()
    plt.savefig(f"figs/{type}_distribution_data.pdf")
    plt.show()
    plt.clf()


def plot_path_manual(ds_name: str, X: np.asarray):
    # the length of the time-series
    sz = X.shape[1]

    path, sim = metrics.dtw_path(X[0], X[1])

    plt.figure(1, figsize=(8, 8))

    # definitions for the axes
    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02

    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]

    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    mat = cdist(X[0], X[1])

    ax_gram.imshow(mat)
    ax_gram.axis("off")
    ax_gram.autoscale(False)
    ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-", linewidth=3.0)

    ax_s_x.plot(np.arange(sz), X[1], "b-", linewidth=3.0)
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, sz - 1))

    ax_s_y.plot(-X[0], np.arange(sz)[::-1], "r-", linewidth=3.0)
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, sz - 1))

    plt.savefig(f"dtw_path.pdf")
    plt.show()
    plt.clf()


def plot_dtw_path(X, y, type: str):
    """Plots DTW Path

    Args:
        type (str): String of data set name
        X (numpy.array): NumPy Arrays
    """
    # Count indices that are greater 0
    x_occ = np.argwhere(X > 0)

    # Get counts
    unique, counts = np.unique(x_occ[:, 0], return_counts=True)
    data = dict(zip(unique, counts))
    results = data.items()

    filterdata = np.array(list(results))

    X = X[np.where(filterdata[:, 1] > 1)]
    y = y[np.where(filterdata[:, 1] > 1)]

    y1 = y[np.where(y == 1)[0]]
    X1 = X[np.where(y == 1)[0]]

    y2 = y[np.where(y == 2)[0]]
    X2 = X[np.where(y == 2)[0]]

    _, paths = dtw.warping_paths(X1[1], X1[0], window=50, use_pruning=True)
    best_path = dtw.best_path(paths)
    fig, _ = dtwvis.plot_warpingpaths(X1[1], X1[0], paths, best_path)
    fig.savefig(f"figs/{type}_dtw_path_class1.pdf")

    _, paths = dtw.warping_paths(X2[1], X2[0], window=50, use_pruning=True)
    best_path = dtw.best_path(paths)
    fig, _ = dtwvis.plot_warpingpaths(X2[1], X2[0], paths, best_path)
    fig.savefig(f"figs/{type}_dtw_path_class2.pdf")
    plt.show()
    plt.clf()


if __name__ == "__main__":
    x_arr = []
    y_arr = []

    data_types = ["cic", "dnscapy", "tuns", "plain"]  # "live", "test"
    data_types = data_types + HEICLOUD_DATA[0]

    # load data
    for data_type in data_types:
        y_arr.append(np.load(f"data/y_{data_type}_1min_entropy.npy"))
        x_arr.append(np.load(f"data/x_{data_type}_1min_entropy.npy"))

    X = np.concatenate(x_arr)
    y = np.concatenate(y_arr)
    y = y.reshape(-1)

    plot_dtw_path(X, y, "cic_new")
    plot_distribution(X, y, "cic_new")
    plot_neighbours(X, y, "cic_new")
