import matplotlib.pyplot as plt
import numpy as np
from tslearn.barycenters import (
    euclidean_barycenter,
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_subgradient,
    softdtw_barycenter,
)

from utils import TIME_INTERVAL_CONFIG, load_dataset


def plot_helper(barycenter):
    # plot all points of the data set
    for series in X:
        plt.plot(series.ravel(), "k-", alpha=0.2)
    # plot the given barycenter of them
    plt.plot(barycenter.ravel(), "r-", linewidth=2)


def plot_dtw_path(X, y, type: str):
    length_of_sequence = X.shape[1]

    # plot the four variants with the same number of iterations and a tolerance of
    # 1e-3 where applicable
    ax1 = plt.subplot(4, 1, 1)

    plt.subplot(4, 1, 2, sharex=ax1)
    plt.title("DBA (vectorized version of Petitjean's EM)")
    plot_helper(dtw_barycenter_averaging(X, max_iter=50, tol=1e-3))

    plt.subplot(4, 1, 3, sharex=ax1)
    plt.title("DBA (subgradient descent approach)")
    plot_helper(dtw_barycenter_averaging_subgradient(X, max_iter=50, tol=1e-3))

    # clip the axes for better readability
    ax1.set_xlim([0, length_of_sequence])

    # show the plot(s)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"figs/cluster_{type}.pdf")

if __name__ == "__main__":
    for ti in TIME_INTERVAL_CONFIG:
        X, y = load_dataset(
            ti["time_interval_name"],
            data=[
                {
                    "name": "cic-malicious",
                    "input_dir": "./dtw_data/cic/attack",
                    "class_type": "1",
                }
            ],
        )

        plot_dtw_path(X, y, ti["time_interval_name"])
