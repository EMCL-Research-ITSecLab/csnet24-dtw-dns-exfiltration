import json
import sys

import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sktime.classification.deep_learning import LSTMFCNClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeriesClassifierTslearn
from sktime.classification.hybrid import HIVECOTEV2
from sktime.dists_kernels import FlatDist, ScipyDist

from utils import (
    HEICLOUD_DATA,
    TIME_INTERVAL_CONFIG,
    TS_TYPE,
    fdr,
    fpr,
    fttar,
    load_dataset,
)


def train(name, clf):
    for ts in TS_TYPE:
        for ti in TIME_INTERVAL_CONFIG:
            result = dict()

            result["time_interval"] = ti["time_interval_name"]
            result["ts_type"] = ts
            result["model"] = name

            print(f"Run analysis on data: {ti['time_interval_name']} for {ts}")

            X, y, _, _ = load_dataset(ti["time_interval_name"], ts_type=ts)

            print(f"Data has shape: {X.shape}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            clf.fit(X_train, y_train)

            joblib.dump(
                clf, f"models/model_{name}_{ti['time_interval_name']}_{ts}.pickle"
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

            # print("Predicting production set...")
            # result["prod"] = {}

            # X_new, y_new, _, _ = load_dataset(
            #     ti["time_interval_name"], ts_type=ts, data=HEICLOUD_DATA
            # )

            # y_pred = clf.predict(X_new)
            # report = classification_report(y_new, y_pred, output_dict=True)
            # fttar_test = fttar(y_new, y_pred)
            # fpr_test = fpr(y_new, y_pred)
            # fdr_test = fdr(y_new, y_pred)

            # result["prod"]["report"] = report
            # result["prod"]["fttar"] = fttar_test
            # result["prod"]["fpr"] = fpr_test
            # result["prod"]["fdr"] = fdr_test

            # print(report)
            # print(f"FTTAR: {fttar_test}")
            # print(f"False Positive Rate: {fpr_test}")
            # print(f"False Discovery Rate: {fdr_test}")

            with open(f"result_{ti['time_interval_name']}.json", "a+") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    name = sys.argv[1]
    match name:
        case "lstm":
            clf = LSTMFCNClassifier(verbose=1, n_epochs=100)
        case "hivecotev":
            clf = HIVECOTEV2(n_jobs=-1, verbose=1)
        case "knn-euclidean":
            eucl_dist = FlatDist(ScipyDist())
            clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, n_jobs=-1, distance=eucl_dist)
        case "knn-dtw":
            clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, n_jobs=-1, distance="dtw")
        case "knn-dtw-sakoe":
            clf = KNeighborsTimeSeriesClassifierTslearn(
                n_neighbors=2,
                verbose=1,
                n_jobs=-1,
                metric="dtw",
                metric_params={
                    "global_constraint": "sakoe_chiba",
                    "sakoe_chiba_radius": 3,
                },
            )
        case "knn-dtw-soft":
            clf = KNeighborsTimeSeriesClassifierTslearn(
                n_neighbors=2,
                verbose=1,
                n_jobs=-1,
                metric="softdtw",
            )
        case "knn-dtw-itakura":
            clf = KNeighborsTimeSeriesClassifierTslearn(
                n_neighbors=2,
                n_jobs=-1,
                verbose=1,
                metric="dtw",
                metric_params={
                    "global_constraint": "itakura",
                    "itakura_max_slope": 2.0,
                },
            )
        case _:
            raise NotImplementedError(f"{name} not found")
    train(name, clf)
