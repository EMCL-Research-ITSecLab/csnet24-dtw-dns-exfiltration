import glob
import pickle
from string import ascii_lowercase as alc

import joblib
import polars as pl
import sklearn
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

from xg import make_ensemble_preds, train_m_models
from xg_dataset import load_data

if __name__ == "__main__":
    ytest = []
    xtest = []
    
    ytrain = []
    xtrain = []
    
    files = glob.glob("./dgarchive/*.csv")
    files.append("/home/smachmeier/projects/heiDGA/full_data.csv")
    for file in files:
        df = pl.read_csv(file, separator=",")[:10000]
        y = df.select("class")
        x = df.drop("class")
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
            x,
            y,
            train_size=0.8,
            random_state=42,
        )
        xtrain.append(X_train)
        ytrain.append(Y_train)
        
        xtrain.append(X_test)
        ytrain.append(Y_test)

    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = train_m_models(xtrain, ytrain)
    
    joblib.dump(models, "models.pickle")
    models = joblib.load("models.pickle")
    
    ensemble_reports, ensemble_cms, ensemble_preds = make_ensemble_preds(
        xtest, ytest, models
    )
    print(ensemble_reports)
