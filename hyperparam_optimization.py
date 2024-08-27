import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifierTslearn

from utils import (
    TIME_INTERVAL_CONFIG,
    TS_TYPE,
    load_dataset,
)

if __name__ == "__main__":
    for ts in TS_TYPE:
        for ti in TIME_INTERVAL_CONFIG:
            X, y, _, _ = load_dataset(ti["time_interval_name"], ts_type=ts)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            parameters = [
                {"n_neighbors": np.arange(2, 16), "weights": ["uniform", "distance"]},
                {
                    "n_neighbors": np.arange(2, 16),
                    "weights": ["uniform", "distance"],
                    "metric_params": [
                        {
                            "global_constraint": "sakoe_chiba",
                            "sakoe_chiba_radius": i,
                        }
                        for i in np.linspace(1, 5, 10, dtype=float)
                    ],
                },
                {
                    "n_neighbors": np.arange(2, 16),
                    "weights": ["uniform", "distance"],
                    "metric_params": [
                        {
                            "global_constraint": "itakura",
                            "itakura_max_slope": i,
                        } for i in np.linspace(1, 5, 10, dtype=float)
                    ],
                },
            ]

            clf = GridSearchCV(
                KNeighborsTimeSeriesClassifierTslearn(
                    metric="dtw",
                    n_jobs=-1,
                ),
                parameters,
                cv=2,
                n_jobs=-1,
                verbose=2,
            )

            clf.fit(X_train, y_train)

            print(clf.best_estimator_)

            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
