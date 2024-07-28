import matplotlib.pyplot as plt
from tslearn.barycenters import (dtw_barycenter_averaging,
                                 dtw_barycenter_averaging_subgradient)

from utils import TIME_INTERVAL_CONFIG, load_dataset


def plot_helper(barycenter):
    # plot all points of the data set
    for series in X:
        plt.plot(series.ravel(), "k-", alpha=0.2)
    # plot the given barycenter of them
    plt.plot(barycenter.ravel(), "r-", linewidth=2)


def plot_dtw_path(x, y, type: str):
    length_of_sequence = x.shape[1]

    # plot the four variants with the same number of iterations and a tolerance of
    # 1e-3 where applicable
    ax1 = plt.subplot(2, 1, 1)

    # plt.subplot(2, 1, 1, sharex=ax1)
    plt.title("DBA (vectorized version of Petitjean's EM)")
    plot_helper(dtw_barycenter_averaging(x, max_iter=50, tol=1e-5))

    plt.subplot(2, 1, 2, sharex=ax1)
    plt.title("DBA (subgradient descent approach)")
    plot_helper(dtw_barycenter_averaging_subgradient(x, max_iter=50, tol=1e-5))

    # clip the axes for better readability
    ax1.set_xlim([0, length_of_sequence])

    # show the plot(s)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"figs/cluster_{type}.pdf")
    ax1.cla()
    plt.clf()

if __name__ == "__main__":
    for ti in TIME_INTERVAL_CONFIG:
        X, y, _, _ = load_dataset(
            ti["time_interval_name"],
            data=[
                {
                    "name": "cic-malicious",
                    "input_dir": "./dtw_data/cic/attack",
                    "class_type": "1",
                }
            ],
        )
        print(X[0:1])
        plot_dtw_path(X[0:1], y, ti["time_interval_name"])
        plt.clf()
