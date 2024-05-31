import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def fit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # train
    parameters = {"n_neighbors": [2, 4, 8, 10, 12, 14, 16]}
    clf = GridSearchCV(
        KNeighborsTimeSeriesClassifier(metric="dtw", n_jobs=-1),
        parameters,
        cv=2,
        n_jobs=-1,
        verbose=2,
    )

    clf.fit(X_train, y_train)
    print(clf.best_params_["n_neighbors"])

    # evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, "dtwknn.pickle")

    # sampl = np.random.uniform(low=120, high=120.5, size=(100,))
    # y_pred = clf.predict([sampl])
    # print(y_pred)


if __name__ == "__main__":
    y = np.load("data/y_cic_1min_entropy.npy")
    X = np.load("data/x_cic_1min_entropy.npy")

    # y1 = np.load("data/y_data_1min_packet_count.npy")
    # X1 = np.load("data/x_data_1min_packet_count.npy")

    y1 = np.load("data/y_test_1min_entropy.npy")
    X1 = np.load("data/x_test_1min_entropy.npy")
    
    y2 = np.load("data/y_testg_1min_entropy.npy")
    X2 = np.load("data/x_testg_1min_entropy.npy")

    # y2 = np.load("data/y_heicloud_1min_packet_count.npy")
    # X2 = np.load("data/x_heicloud_1min_packet_count.npy")

    X = np.concatenate([X, X1, X2])
    y = np.concatenate([y, y1, y2])

    # Count indices that are greater 0
    x_occ = np.argwhere(X > 0)
    # Get counts
    unique, counts = np.unique(x_occ[:, 0], return_counts=True)
    data = dict(zip(unique, counts))
    results = data.items()
    filterdata = np.array(list(results))

    X = X[np.where(filterdata[:, 1] > 1)]
    y = y[np.where(filterdata[:, 1] > 1)]

    y2 = y[np.where(y == 2)[0]]
    X2 = X[np.where(y == 2)[0]]

    y1 = y[np.where(y == 1)[0]][:1000]
    X1 = X[np.where(y == 1)[0]][:1000]

    X = np.concatenate([X1, X2])
    y = np.concatenate([y1, y2])

    # scaler = TimeSeriesScalerMeanVariance()  # Rescale time series
    # X = scaler.fit_transform(X)
    y = y.reshape(-1)

    fit(X, y)
