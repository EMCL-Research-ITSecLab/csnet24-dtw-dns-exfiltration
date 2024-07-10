import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sktime.classification.deep_learning import LSTMFCNClassifier
from sktime.classification.hybrid import HIVECOTEV2

from dtw_utils import HEICLOUD_DATA


if __name__ == "__main__":


    time_interval_names = ["15s", "30s", "1min", "5min", "1h", "6h", "12h", "24h"]
    data_types = ["cic"]
    
    for time_interval_name in time_interval_names:
        x_arr = []
        y_arr = []
        # load data
        for data_type in data_types:
            y_arr.append(np.load(f"dtw_data_npy/y_{data_type}_{time_interval_name}_entropy.npy"))
            x_arr.append(np.load(f"dtw_data_npy/x_{data_type}_{time_interval_name}_entropy.npy"))

        X = np.concatenate(x_arr)
        y = np.concatenate(y_arr)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        
        # TODO Hyperparam optimization
        clf = LSTMFCNClassifier(verbose=1, n_epochs=10, lstm_size=12)
        clf.fit(X_train, y_train)
        
        joblib.dump(clf, f"models/sktime_lstm_{time_interval_name}.pickle")
        
        print("Predicting test set...")
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        # TODO better metrics for results
        
        print("Predicting production set...")
        x_arr_new = []
        y_arr_new = []
        for data_type in HEICLOUD_DATA:
            y_arr_new.append(np.load(f"dtw_data_npy/y_{data_type}_{time_interval_name}_entropy.npy"))
            x_arr_new.append(np.load(f"dtw_data_npy/x_{data_type}_{time_interval_name}_entropy.npy"))

        X_new = np.concatenate(x_arr_new)
        y_new = np.concatenate(y_arr_new)
        
        y_pred = clf.predict(X_new)
        print(classification_report(y_new, y_pred))
        