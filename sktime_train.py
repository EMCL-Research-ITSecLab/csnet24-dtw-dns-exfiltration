import json
import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sktime.classification.deep_learning import LSTMFCNClassifier

from utils import DATA_CONFIG, HEICLOUD_DATA, TIME_INTERVAL_CONFIG, fdr, fpr, fttar

def train(model):
    ts_type = ["univariate", "mutlivariate"]

    for ts_type in ts_type:
        for ti in TIME_INTERVAL_CONFIG:
            result = dict()
            
            result["time_interval"] = ti['time_interval_name']
            result["ts_type"] = ts_type
            
            print(f"Run analysis on data: {ti['time_interval_name']} for {ts_type}")
            
            x_arr = []
            y_arr = []

            # load data
            for data_type in DATA_CONFIG:
                y = np.load(f"dtw_data_npy/y_{data_type['name']}_{ti['time_interval_name']}_entropy.npy")
                
                if ts_type == "mutlivariate":
                    X = np.stack([np.load(f"dtw_data_npy/x_{data_type['name']}_{ti['time_interval_name']}_entropy.npy"),np.load(f"dtw_data_npy/x_{data_type['name']}_{ti['time_interval_name']}_packet_size.npy")], axis=1)
                else: 
                    X = np.load(f"dtw_data_npy/x_{data_type['name']}_{ti['time_interval_name']}_entropy.npy")
                if X.size != 0:
                    y_arr.append(y)
                    x_arr.append(X)
                else:
                    print(f"WARNING: {data_type['name']} for {ti['time_interval_name']} is empty!")

            X = np.concatenate(x_arr)
            y = np.concatenate(y_arr)
            
            print(f"Data has shape: {X.shape}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.30, random_state=42, stratify=y
            )
            
            clf = LSTMFCNClassifier(verbose=0, n_epochs=10, lstm_size=12)
            clf.fit(X_train, y_train)
            
            joblib.dump(clf, f"models/sktime_lstm_{ti['time_interval_name']}_{ts_type}.pickle")
            
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
            
            x_arr_new = []
            y_arr_new = []
            
            for data_type in HEICLOUD_DATA:
                y_arr.append(np.load(f"dtw_data_npy/y_{data_type}_{ti['time_interval_name']}_entropy.npy"))
                if ts_type == "mutlivariate":
                    X = np.stack([np.load(f"dtw_data_npy/x_{data_type}_{ti['time_interval_name']}_entropy.npy"),np.load(f"dtw_data_npy/x_{data_type}_{ti['time_interval_name']}_packet_size.npy")], axis=2)
                else: 
                    X = np.load(f"dtw_data_npy/x_{data_type}_{ti['time_interval_name']}_entropy.npy")
                x_arr.append(X)

            X_new = np.concatenate(x_arr_new)
            y_new = np.concatenate(y_arr_new)
            
            y_pred = clf.predict(X_new)
            report = classification_report(y_test, y_pred, output_dict=True)
            fttar_test = fttar(y_test, y_pred)
            fpr_test = fpr(y_test, y_pred)
            fdr_test = fdr(y_test, y_pred)

            result["prod"]["report"] = report
            result["prod"]["fttar"] = fttar_test
            result["prod"]["fpr"] = fpr_test
            result["prod"]["fdr"] = fdr_test
            
            print(report)
            print(f"FTTAR: {fttar_test}")
            print(f"False Positive Rate: {fpr_test}")
            print(f"False Discovery Rate: {fdr_test}")
            
            with open("result_lstm.json", "a") as f:
                f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    