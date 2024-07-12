import matplotlib.pyplot as plt
import numpy as np
from tslearn import metrics
from scipy.spatial.distance import cdist

from utils import TIME_INTERVAL_CONFIG, load_dataset


def plot_dtw_path(X, y, type: str):
    y1 = y[np.where(y == 1)[0]]
    X1 = X[np.where(y == 1)[0]]

    y2 = y[np.where(y == 2)[0]]
    X2 = X[np.where(y == 2)[0]]

    s_y1 = X1[1].reshape((-1, 1))
    s_y2 = X1[0].reshape((-1, 1))
    sz = s_y1.shape[0]
    
    # path, sim = metrics.soft_dtw_alignment(s_y1, s_y2,gamma=0.1)
    path, sim = metrics.dtw_path(s_y1, s_y2, global_constraint="itakura", itakura_max_slope=2.)
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

    mat = cdist(s_y1, s_y2)

    ax_gram.imshow(mat, origin='lower')
    ax_gram.axis("off")
    ax_gram.autoscale(False)
    ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-",
                linewidth=3.)

    ax_s_x.plot(np.arange(sz), s_y2, "b-", linewidth=3.)
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, sz - 1))

    ax_s_y.plot(- s_y1, np.arange(sz), "b-", linewidth=3.)
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, sz - 1))

    plt.title(f"DTW Path \nfor {type}\ndistance = {round(sim,2)}")
    plt.tight_layout()
    plt.savefig(f"figs/{type}_dtw_path_class.pdf")
    plt.show()
    plt.clf()


if __name__ == "__main__":
    for ti in TIME_INTERVAL_CONFIG:
        X, y = load_dataset(ti["time_interval_name"])

        plot_dtw_path(X, y, ti["time_interval_name"])
