import json
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sktime.classification.deep_learning import LSTMFCNClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.dists_kernels import FlatDist, ScipyDist

from utils import HEICLOUD_DATA, TIME_INTERVAL_CONFIG, TS_TYPE, fdr, fpr, fttar, load_dataset


def train(name, model):
    for ts in TS_TYPE:
        for ti in TIME_INTERVAL_CONFIG:
            result = dict()

            result["time_interval"] = ti["time_interval_name"]
            result["ts_type"] = ts

            print(f"Run analysis on data: {ti['time_interval_name']} for {ts}")

            X, y = load_dataset(ti["time_interval_name"], ts_type=ts)

            print(f"Data has shape: {X.shape}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # TODO Train model + hyperparam.
            clf = model(verbose=0, n_epochs=10, lstm_size=12)
            clf.fit(X_train, y_train)

            joblib.dump(
                clf, f"models/sktime_lstm_{ti['time_interval_name']}_{ts}.pickle"
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

            with open(f"result_{name}.json", "a") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    train(model=LSTMFCNClassifier)
    # TODO How to train?
    # eucl_dist = FlatDist(ScipyDist())
    # clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, distance=eucl_dist)

    # HIVECOTEV2(n_jobs=-1, verbose=1)
