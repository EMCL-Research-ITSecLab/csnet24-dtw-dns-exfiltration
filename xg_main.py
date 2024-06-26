import glob
import pickle
from string import ascii_lowercase as alc

import joblib
import numpy as np
import polars as pl
import sklearn
import torch
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import OneClassSVM

from xg import make_ensemble_preds, train_m_models
from xg_dataset import load_data

if __name__ == "__main__":
    ytest = []
    xtest = []

    ytrain = []
    xtrain = []

    encoder = LabelEncoder()

    files = glob.glob("./dgarchive/*.csv")
    # Load default data set as base for training
    df_default = pl.read_csv(
        "/home/smachmeier/projects/heiDGA/full_data.csv", separator=","
    ).with_columns(pl.all().exclude("class").cast(pl.Float64, strict=False))
    # Sort columns
    df_default = df_default.select(sorted(df_default.columns))
    for file in files:
        df = pl.read_csv(file, separator=",", n_rows=1000000).with_columns(
            pl.all().exclude("class").cast(pl.Float64, strict=False)
        )
        df = df.select(sorted(df.columns))
        df = pl.concat(
            [
                df,
                df_default.sample(
                    n=100000, seed=42, shuffle=True, with_replacement=True
                ),
            ]
        )

        y = df.select("class").to_numpy()
        x = df.drop("class").to_numpy()

        y = encoder.fit_transform(y.reshape(-1))

        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
            x,
            y,
            train_size=0.8,
            random_state=42,
        )

        xtrain.append(X_train)
        ytrain.append(Y_train)

        xtest.append(X_test)
        ytest.append(Y_test)

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_m_models(xtrain, ytrain)

    models = []
    

    xtest = [np.concatenate(xtest)]
    ytest = [np.concatenate(ytest)]
    ensemble_reports, ensemble_cms, ensemble_preds = make_ensemble_preds(
        xtest, ytest, models
    )
    print(ensemble_reports)
