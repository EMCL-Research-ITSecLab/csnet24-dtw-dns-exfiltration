import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from utils import TIME_INTERVAL_CONFIG, TS_TYPE, load_dataset

if __name__ == "__main__":
    for ts in TS_TYPE:
        for ti in TIME_INTERVAL_CONFIG:
            x_arr = []
            y_arr = []

            X, y = load_dataset(ti["time_interval_name"], ts_type=ts)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
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

            joblib.dump(clf.best_estimator_, f"models/dtw_{ti["time_interval_name"]}_{ts}.pickle")

            # evaluate
            y_pred = clf.predict(X_test)

            print(classification_report(y_test, y_pred))
