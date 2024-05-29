import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from xg_dataset import load_data
import polars as pl
import torch
from string import ascii_lowercase as alc
import pickle

from xg import make_ensemble_preds, train_m_models

if __name__ == "__main__":
    # load_data()
    df = pl.read_csv("/home/smachmeier/projects/heiDGA/full_data.csv", separator=",")
    y = df.select("class")
    ytest = []
    xtest = []

    # for level in ["fqdn", "thirdleveldomain", "secondleveldomain"]:
    #     xtest.append(df.select([
    #         f"{level}_full_count",
    #         f"{level}_alpha_count",
    #         f"{level}_numeric_count",
    #         f"{level}_special_count",
    #         f"{level}_median",
    #         f"{level}_var",
    #         f"{level}_std",
    #         f"{level}_mean",
    #         f"{level}_entropy",
    #     ]))
    #     ytest.append(y)

    # freq = [f"freq_{i}" for i in alc]
    # freq.append(f"freq_median")
    # freq.append(f"freq_var",)
    # freq.append(f"freq_std",)
    # freq.append(f"freq_mean",)

    # xtest.append(df.select(freq))
    xtest.append(df.drop("class"))
    ytest.append(y)

    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = train_m_models(xtest, ytest)
    joblib.dump(models, "models.pickle")
    models = joblib.load("models.pickle")
    ensemble_reports, ensemble_cms, ensemble_preds = make_ensemble_preds(
        xtest, ytest, models
    )
    print(ensemble_reports)

    # X = df.drop([
    #                 "query",
    #                 "labels",
    #                 "thirdleveldomain",
    #                 "secondleveldomain",
    #                 "fqdn",
    #                 "tld",
    #                 "class"
    #             ])
    # y = df.select("class")

    # trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
    # model.fit(trainX)
    # y_pred = model.predict(testX)
    # print(y_pred)
    # print(classification_report(testy, y_pred, labels=[1]))
