import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

if __name__ == "__main__":

    time_interval_names = ["15s", "30s", "1min", "5min", "1h", "6h", "12h", "24h"]
    data_types = ["cic"]
    
    for time_interval_name in time_interval_names:
        x_arr = []
        y_arr = []
        # load data
        for data_type in data_types:
            y_arr.append(
                np.load(f"dtw_data_npy/y_{data_type}_{time_interval_name}_entropy.npy")
            )
            x_arr.append(
                np.load(f"dtw_data_npy/x_{data_type}_{time_interval_name}_entropy.npy")
            )

        X = np.concatenate(x_arr)
        y = np.concatenate(y_arr)

        y = y.reshape(-1)

        print(X.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        # train
        parameters = [
            {"n_neighbors": np.arange(2, 16), "weights": ["uniform", "distance"]},
            {
                "n_neighbors": np.arange(2, 16),
                "weights": ["uniform", "distance"],
                "metric_params": {
                    "global_constraint": "itakura",
                    "itakura_max_slope": np.linspace(1, 5, 10, dtype=int),
                },
            },
            {
                "n_neighbors": np.arange(2, 16),
                "weights": ["uniform", "distance"],
                "metric_params": {
                    "global_constraint": "sakoe_chiba",
                    "sakoe_chiba_radius": np.linspace(1, 5, 10, dtype=int),
                },
            },
        ]
        clf = GridSearchCV(
            KNeighborsTimeSeriesClassifier(
                metric="dtw",
                n_jobs=-1,
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
