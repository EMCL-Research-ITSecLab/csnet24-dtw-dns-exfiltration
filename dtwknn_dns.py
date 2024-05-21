import sys
import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import squareform
import polars as pl
import sklearn.model_selection
from sklearn import preprocessing


class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN

    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function

    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """

    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """

        self.x = x
        self.l = l

    def _dtw_distance(self, ts_a, ts_b, d=lambda x, y: abs(x - y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared

        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function

        Returns
        -------
        DTW distance between A and B
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(
                max(1, i - self.max_warping_window), min(N, i + self.max_warping_window)
            ):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window
        return cost[-1, -1]

    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """

        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if np.array_equal(x, y):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(
                        x[i, :: self.subsample_step], y[j, :: self.subsample_step]
                    )

                    dm_count += 1

            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            dm_size = x_s[0] * y_s[0]

            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(
                        x[i, :: self.subsample_step], y[j, :: self.subsample_step]
                    )
                    # Update progress bar
                    dm_count += 1

            return dm

    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels
              (2) the knn label count probability
        """

        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, : self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()


from itertools import pairwise


def multidimensional_to_numpy(s):
    dimensions = [1, len(s)]
    while s.dtype == pl.List:
        s = s.explode()
        dimensions.append(len(s))
    dimensions = [p[1] // p[0] for p in pairwise(dimensions)]
    return s.to_numpy().reshape(dimensions)


if __name__ == "__main__":

    df = pl.read_csv(
        "/home/smachmeier/Downloads/dataset.csv", has_header=False
    ).with_columns([(pl.from_epoch("column_3", time_unit="ms"))])

    print(
        df.select(["column_1", "column_4", "column_5", "column_16"])
        .filter(pl.col("column_4") == True)
        .group_by("column_1")
        .agg(pl.col("column_5").count())
    )

    x = df.select(["column_1", "column_5", "column_3", "column_4"])

    x = x.with_columns(
        [
            pl.col(i).rank("dense").cast(pl.Int64).name.suffix("_encoded")
            for i in ["column_1", "column_5", "column_4"]
        ]
    )
    x = x.drop(["column_1", "column_5"])

    x: pl.DataFrame = (
        x.sort("column_3")
        .group_by_dynamic(
            "column_3",
            every="1h",
            closed="right",
            by=["column_1_encoded", "column_5_encoded", "column_4_encoded"],
        )
        .agg(pl.count())
    )

    min_date = x.select(["column_3"]).min().item()
    max_date = x.select(["column_3"]).max().item()

    # We generate empty datetime with zero values in a time range of 6h
    datetimes = x.select(
        pl.datetime_range(
            min_date.replace(microsecond=0),
            max_date.replace(microsecond=0),
            "1h",
            time_unit="ms",
        ).alias("column_3")
    )

    ids = x.select(
        ["column_1_encoded", "column_5_encoded", "column_4_encoded"]
    ).unique()

    # Cross joining all domain
    all_dates = datetimes.join(ids, how="cross")
    # Fill with null
    x = all_dates.join(
        x,
        how="left",
        on=["column_1_encoded", "column_5_encoded", "column_4_encoded", "column_3"],
    ).fill_null(0)

    x = x.group_by(["column_1_encoded", "column_5_encoded", "column_4_encoded"]).agg(
        pl.col("count")
    )

    Y = x.select(["column_4_encoded"])
    x = x.select(["count"])

    x = multidimensional_to_numpy(x["count"])
    np.save("x_data_h_dest.npy", x)
    Y = Y.to_numpy()
    np.save("y_data_h_dest.npy", Y)

    # Y = np.load('y_data_h.npy')
    # x = np.load('x_data_h.npy')

    # print(x.shape)
    # print(Y.shape)

    # scaler = preprocessing.MinMaxScaler()
    # x = scaler.fit_transform(x)

    # print(x.shape)
    # print(Y.shape)

    # # Count indices that are greater 0
    # x_occ = np.argwhere(x > 0)
    # # Get counts
    # unique, counts = np.unique(x_occ[:,1], return_counts=True)
    # data = dict(zip(unique, counts))
    # results = data.items()
    # filterdata = np.array(list(results))

    # x = x[np.where(filterdata[:,1] > 27)]
    # Y = Y[np.where(filterdata[:,1] > 27)]

    # Y = Y.reshape(-1)

    # x_train, x_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    #     x,
    #     Y,
    #     train_size=0.7,
    #     random_state=0,
    # )

    # plt.figure(figsize=(11,7))
    # colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
    #         '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

    # labels = {1:'Benign', 2:'Malicious'}

    # for i, r in enumerate([0,27,65,100,145,172]):
    #     plt.subplot(3,2,i+1)
    #     plt.plot(x_train[r], label=labels[Y_train[r]], color=colors[i], linewidth=2)
    #     plt.xlabel('Samples @50Hz')
    #     plt.legend(loc='upper left')
    #     plt.tight_layout()
    # plt.savefig("data.pdf")

    # m = KnnDtw(n_neighbors=1, max_warping_window=10)

    # m.fit(x_train, Y_train)
    # label, proba = m.predict(x_test)

    # print(label)
    # from sklearn.metrics import classification_report, confusion_matrix
    # print(classification_report(label, Y_test, target_names=[l for l in labels.values()]))

    # conf_mat = confusion_matrix(label, Y_test)

    # fig = plt.figure(figsize=(6,6))
    # width = np.shape(conf_mat)[1]
    # height = np.shape(conf_mat)[0]

    # res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    # for i, row in enumerate(conf_mat):
    #     for j, c in enumerate(row):
    #         if c>0:
    #             plt.text(j-.2, i+.1, c, fontsize=16)

    # cb = fig.colorbar(res)
    # plt.title('Confusion Matrix')
    # _ = plt.xticks(range(6), [l for l in labels.values()], rotation=90)
    # _ = plt.yticks(range(6), [l for l in labels.values()])
    # plt.savefig("cm.pdf")
