import math
import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from pyclustering.nnet.som import type_conn
from sklearn.metrics import classification_report
from tslearn.clustering import TimeSeriesKMeans
import dtwsom
from tslearn.barycenters import \
    euclidean_barycenter, \
    dtw_barycenter_averaging, \
    dtw_barycenter_averaging_subgradient, \
    softdtw_barycenter

from utils import TIME_INTERVAL_CONFIG, load_dataset
def plot_helper(barycenter):
    # plot all points of the data set
    for series in X:
        plt.plot(series.ravel(), "k-", alpha=.2)
    # plot the given barycenter of them
    plt.plot(barycenter.ravel(), "r-", linewidth=2)



# Little handy function to plot series
def plot_som_series_averaged_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
    fig.suptitle('Clusters')
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series,c="gray",alpha=0.5) 
                axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
            cluster_number = x*som_y+y+1
            axs[cluster].set_title(f"Cluster {cluster_number}")

    plt.show()


def plot_dtw_path(X, y, type: str):
    # y1 = y[np.where(y == 1)[0]]
    # X = X[np.where(y == 2)[0]]

    # y2 = y[np.where(y == 2)[0]]
    # X2 = X[np.where(y == 2)[0]]
    
    rows = 3
    cols = 3
    structure = type_conn.grid_four
    
    som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(X))))
    network = dtwsom.DtwSom(som_x, som_y, structure)
    network.train(X, 20)
    plt.tight_layout()
    network.show_distance_matrix()
    plt.savefig("som.pdf")
    

if __name__ == "__main__":
    for ti in TIME_INTERVAL_CONFIG:
        X, y = load_dataset(ti["time_interval_name"])

        plot_dtw_path(X[:100], y, ti["time_interval_name"])

        break