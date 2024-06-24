import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def fit(X, y, model_name = "dtwknn"):
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

    joblib.dump(clf, f"{model_name}.pickle")

    # sampl = np.random.uniform(low=120, high=120.5, size=(100,))
    # y_pred = clf.predict([sampl])
    # print(y_pred)


if __name__ == "__main__":
    x_arr = []
    y_arr = []
    
    # load data
    for data in ["cic", "live"]:
        y_arr.append(np.load(f"data/y_{data}_1min_entropy.npy"))
        x_arr.append(np.load(f"data/x_{data}_1min_entropy.npy"))
        

    X = np.concatenate(x_arr)
    y = np.concatenate(y_arr)

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

    fit(X, y, model_name="dtwknn_test")
