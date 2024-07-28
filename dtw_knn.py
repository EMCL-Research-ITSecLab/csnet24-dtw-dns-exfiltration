import json

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from utils import (
    HEICLOUD_DATA,
    TIME_INTERVAL_CONFIG,
    TS_TYPE,
    fdr,
    fpr,
    fttar,
    load_dataset,
)

if __name__ == "__main__":
    for ts in TS_TYPE:
        for ti in TIME_INTERVAL_CONFIG:
            result = dict()

            X, y, _, _ = load_dataset(ti["time_interval_name"], ts_type=ts)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # train
            parameters = [
                {"n_neighbors": np.arange(2, 16), "weights": ["uniform", "distance"]},
                # {
                #     "n_neighbors": np.arange(2, 16),
                #     "weights": ["uniform", "distance"],
                #     "metric_params": {
                #         "global_constraint": "itakura",
                #         "itakura_max_slope": np.linspace(1, 5, 10, dtype=int),
                #     },
                # },
                # {
                #     "n_neighbors": np.arange(2, 16),
                #     "weights": ["uniform", "distance"],
                #     "metric_params": {
                #         "global_constraint": "sakoe_chiba",
                #         "sakoe_chiba_radius": np.linspace(1, 5, 10, dtype=int),
                #     },
                # },
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

            joblib.dump(
                clf.best_estimator_,
                f"models/dtw_{ti['time_interval_name']}_{ts}.pickle",
            )

            print("Predicting test set...")
            result["test"] = {}
            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            fttar_test = fttar(y_test, y_pred)
            fpr_test = fpr(y_test, y_pred)
            fdr_test = fdr(y_test, y_pred)

            result["test"]["report"] = report
            result["test"]["fttar"] = fttar_test
            result["test"]["fpr"] = fpr_test
            result["test"]["fdr"] = fdr_test

            print(report)
            print(f"FTTAR: {fttar_test}")
            print(f"False Positive Rate: {fpr_test}")
            print(f"False Discovery Rate: {fdr_test}")

            print("Predicting production set...")
            result["prod"] = {}

            X_new, y_new = load_dataset(
                ti["time_interval_name"], ts_type=ts, data=HEICLOUD_DATA
            )

            y_pred = clf.predict(X_new)
            report = classification_report(y_new, y_pred, output_dict=True)
            fttar_test = fttar(y_new, y_pred)
            fpr_test = fpr(y_new, y_pred)
            fdr_test = fdr(y_new, y_pred)

            result["prod"]["report"] = report
            result["prod"]["fttar"] = fttar_test
            result["prod"]["fpr"] = fpr_test
            result["prod"]["fdr"] = fdr_test

            print(report)
            print(f"FTTAR: {fttar_test}")
            print(f"False Positive Rate: {fpr_test}")
            print(f"False Discovery Rate: {fdr_test}")

            with open(f"result_dtw.json", "a") as f:
                f.write(json.dumps(result) + "\n")
