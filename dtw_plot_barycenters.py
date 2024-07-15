import math
import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from sklearn.metrics import classification_report
from tslearn.clustering import TimeSeriesKMeans
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



def plot_som_series_dba_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
    fig.suptitle('Clusters')
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series,c="gray",alpha=0.5) 
                axs[cluster].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red") # I changed this part
            cluster_number = x*som_y+y+1
            axs[cluster].set_title(f"Cluster {cluster_number}")

    plt.show()



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
    plt.savefig("test.pdf")


def plot_dtw_path(X, y, type: str):
    # y1 = y[np.where(y == 1)[0]]
    X = X[np.where(y == 2)[0]]

    # y2 = y[np.where(y == 2)[0]]
    # X2 = X[np.where(y == 2)[0]]

    length_of_sequence = X.shape[1]
    som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(X))))
    # I didn't see its significance but to make the map square,
    # I calculated square root of map size which is 
    # the square root of the number of series
    # for the row and column counts of som

    som = MiniSom(som_x, som_y,len(X[0]), sigma=0.1, learning_rate = 0.001)

    som.random_weights_init(X)
    som.train(X, 50000, use_epochs=True)

    win_map = som.win_map(X)
    # Returns the mapping of the winner nodes and inputs

    # plot_som_series_averaged_center(som_x, som_y, win_map)
    plot_som_series_dba_center(som_x, som_y, win_map)

if __name__ == "__main__":
    for ti in TIME_INTERVAL_CONFIG:
        X, y = load_dataset(ti["time_interval_name"])
        # X = X[np.where(y == 1)[0]]
        # y = y[np.where(y == 1)[0]]
         
        cluster_count = math.ceil(math.sqrt(len(X))) 
        print(cluster_count)
        plot_dtw_path(X, y, ti["time_interval_name"])
        # km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")

        # labels = km.fit_predict(X)
        # print(classification_report(y, labels))
        
        # plot_count = math.ceil(math.sqrt(cluster_count))

        # som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(X))))
        # fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
        # fig.suptitle('Clusters')
        # row_i=0
        # column_j=0
        # # For each label there is,
        # # plots every series with that label
        # for label in set(labels):
        #     cluster = []
        #     for i in range(len(labels)):
        #             if(labels[i]==label):
        #                 axs[row_i, column_j].plot(X[i],c="gray",alpha=0.4)
        #                 cluster.append(X[i])
        #     if len(cluster) > 0:
        #         axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
        #     axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
        #     column_j+=1
        #     if column_j%plot_count == 0:
        #         row_i+=1
        #         column_j=0
                
        # plt.show()
        break