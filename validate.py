import json
import sys

import joblib
from sklearn.metrics import classification_report

from utils import (
    HEICLOUD_DATA,
    TIME_INTERVAL_CONFIG,
    TS_TYPE,
    fdr,
    fpr,
    fttar,
    load_dataset,
)


def validate(name):
    for ts in TS_TYPE:
        for ti in TIME_INTERVAL_CONFIG:
            
            clf = joblib.load(
                f"/mnt/data/models/model_{name}_{ti['time_interval_name']}_{ts}.pickle"
            )
            
            for i in len(HEICLOUD_DATA):
                result = dict()

                result["time_interval"] = ti["time_interval_name"]
                result["ts_type"] = ts
                result["model"] = name
                result["day"] = HEICLOUD_DATA[i]

                print("Predicting production set...")
                result["prod"] = {}
                
                X_new, y_new, _, _ = load_dataset(
                    ti["time_interval_name"], ts_type=ts, data=[HEICLOUD_DATA[i]]
                )
                
                print(f"Data has shape: {X_new.shape}")

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

                with open(f"result_{ti['time_interval_name']}.json", "a+") as f:
                    f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    name = sys.argv[1]
    validate(name)
