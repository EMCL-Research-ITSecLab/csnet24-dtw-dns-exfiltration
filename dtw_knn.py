import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from tslearn.neighbors import KNeighborsTimeSeriesClassifier


def fit(X, y, model_name="dtwknn"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # train
    parameters = {"n_neighbors": [2, 4, 8, 10, 12, 14, 16]} # , "weights": ["uniform", "distance"]
    clf = GridSearchCV(
        KNeighborsTimeSeriesClassifier(
            metric="dtw", n_jobs=-1 #, metric_params={"global_constraint": "itakura", "itakura_max_slope": 1.}
        ),
        parameters,
        cv=2,
        n_jobs=-1,
        verbose=2,
    )

    clf.fit(X_train, y_train)
    print(clf.best_params_)
    joblib.dump(clf, f"models/{model_name}.pickle")

    # evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    x_arr = []
    y_arr = []

    time_interval_name = "15s"
    data_types = [ "cic"]  # "dnscapy", "tuns", "plain", , "live", "test"
    # data_types = data_types + HEICLOUD_DATA

    # load data
    for data_type in data_types:
        y_arr.append(np.load(f"dtw_data_npy/y_{data_type}_{time_interval_name}_entropy.npy"))
        x_arr.append(np.load(f"dtw_data_npy/x_{data_type}_{time_interval_name}_entropy.npy"))

    X = np.concatenate(x_arr)
    y = np.concatenate(y_arr)

    # # Count indices that are greater 0
    # x_occ = np.argwhere(X > 0)
    # # Get counts
    # unique, counts = np.unique(x_occ[:, 0], return_counts=True)
    # data = dict(zip(unique, counts))
    # results = data.items()
    # filterdata = np.array(list(results))

    # X = X[np.where(filterdata[:, 1] > 2)]
    # y = y[np.where(filterdata[:, 1] > 2)]

    # y2 = y[np.where(y == 2)[0]][:1000]
    # X2 = X[np.where(y == 2)[0]][:1000]

    # y1 = y[np.where(y == 1)[0]][:1000]
    # X1 = X[np.where(y == 1)[0]][:1000]

    # X = np.concatenate([X1, X2])
    # y = np.concatenate([y1, y2])

    y = y.reshape(-1)
    
    print(X.shape)

    fit(X, y, model_name="dtwknn_test_2")
