import math

import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans

from utils import TIME_INTERVAL_CONFIG, load_dataset


def plot_som_series_dba_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x, som_y, figsize=(25, 25))
    fig.suptitle("Clusters")
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x, y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series, c="gray", alpha=0.5)
                axs[cluster].plot(
                    dtw_barycenter_averaging(np.vstack(win_map[cluster])), c="red"
                )  # I changed this part
            cluster_number = x * som_y + y + 1
            axs[cluster].set_title(f"Cluster {cluster_number}")

    plt.show()


# Little handy function to plot series
def plot_som_series_averaged_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x, som_y, figsize=(25, 25))
    fig.suptitle("Clusters")
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x, y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series, c="gray", alpha=0.5)
                axs[cluster].plot(
                    np.average(np.vstack(win_map[cluster]), axis=0), c="red"
                )
            cluster_number = x * som_y + y + 1
            axs[cluster].set_title(f"Cluster {cluster_number}")

    plt.show()
    plt.savefig("test.pdf")


def plot_dtw_path(X, y, type: str):
    som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(X))))

    som = MiniSom(som_x, som_y, len(X[0]), sigma=0.3, learning_rate=0.1)
    som.random_weights_init(X)
    som.train(X, 50000)
    win_map = som.win_map(X)

    plot_som_series_averaged_center(som_x, som_y, win_map)
    plot_som_series_dba_center(som_x, som_y, win_map)


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

        cluster_count = math.ceil(math.sqrt(len(X)))
        
        plot_dtw_path(X, y, ti["time_interval_name"])

        km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")

        labels = km.fit_predict(X)

        plot_count = math.ceil(math.sqrt(cluster_count))

        som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(X))))
        fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
        fig.suptitle("Clusters")
        row_i = 0
        column_j = 0
        # For each label there is,
        # plots every series with that label
        for label in set(labels):
            cluster = []
            for i in range(len(labels)):
                if labels[i] == label:
                    axs[row_i, column_j].plot(X[i], c="gray", alpha=0.4)
                    cluster.append(X[i])
            if len(cluster) > 0:
                axs[row_i, column_j].plot(
                    np.average(np.vstack(cluster), axis=0), c="red"
                )
            axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j))
            column_j += 1
            if column_j % plot_count == 0:
                row_i += 1
                column_j = 0

        plt.show()
        plt.savefig("figs/cluster_cic.pdf")
        break
