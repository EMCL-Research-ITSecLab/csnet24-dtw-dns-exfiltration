import pickle
from collections import Counter

import joblib
import numpy as np
import optuna
import sklearn
import sklearn.metrics
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import OneClassSVM
from sklearn.utils import class_weight

# from utils.plots import *


def test_models(models, xs_test, ys_test):
    """
    Trains m models individually on m data sets
    Args:
        models (List): List of XBoost.classifier.
        xs_test (numpy.ndarray): Test matrix.
        ys_test (numpy.ndarray): Test labels.

    Returns:
        list: A list containing dictionaries of classifiction reports.
        list: A list of m confusion matrices.
    """

    reports = []
    cms = []
    y_preds = []
    for xgb_classifier, X, y in zip(models, xs_test, ys_test):
        y_pred = xgb_classifier.predict(X)
        report, cm = test(y, y_pred)
        reports.append(report)
        cms.append(cm)
        y_preds.append(y_pred)

    return reports, cms


def test(y, y_pred):
    """
    Evaluates a classification prediction.

    Args:
        y (pandas.Series or numpy.ndarray): The true labels.
        y_pred (pandas.Series or numpy.ndarray): The predicted labels.

    Returns:
        dict: A dictionary containing the classification report.
        numpy.ndarray: The confusion matrix.
    """
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    return report, cm


def make_ensemble_preds(xs_test, ys_test, models):
    """
    Trains m models individually on m data sets
    Args:
        xs_test (numpy.ndarray): Test matrix.
        ys_test (numpy.ndarray): Test labels.
        models (list): m xgboost.XGBClassifier.

    Returns:
        list: A list containing dictionaries of classifiction reports.
        list: A list of m confusion matrices.
        list: A list of predictions of m data sets.
    """

    ensemble_cms = []
    ensemble_reports = []
    ensemble_preds = []
    y_preds = []
    for index, (X, y) in enumerate(zip(xs_test, ys_test)):
        # for model in models:
        probas = models[index].predict_proba(X)
        y_preds.append(probas)

    ensemble_pred = majority_vote(y_preds, rule="soft")
    ensemble_pred, probas = get_index_and_proba(ensemble_pred)
    maj_report, cm = test(y=y, y_pred=ensemble_pred)
    ensemble_reports.append(maj_report)
    ensemble_cms.append(cm)
    ensemble_preds.append(ensemble_pred)

    return ensemble_reports, ensemble_cms, ensemble_preds


def train_m_models(xs_train, ys_train):
    """
    Trains m XGBoost models on m data sets
    Args:
        xs_train (list): list of train matrices.
        ys_train (list): list of train labels.
    Returns:
        list: A list containing m xgboost.XGBClassifier models.
    """
    i = 0
    for X, y in zip(xs_train, ys_train):
        model = xgb.XGBClassifier(
            tree_method="hist", device="cuda", verbosity=0, silent=True
        )
        model = hypertrain(model, X, y)
        joblib.dump(model, f"models/xgboost_model_{i}.pickle")
        i += 1


def hypertrain(model, x, y):
    """
    Trains a model on the provided features (x) and labels (y).

    Args:
        model (sklearn classifier): Model that is trained.
        x (pandas.DataFrame or numpy.ndarray): The features used for training.
        y (pandas.Series or numpy.ndarray): The labels used for training.

    Returns:
        xgboost.XGBClassifier: The trained model.
    """

    classes_weights = class_weight.compute_sample_weight(class_weight="balanced", y=y)

    hp_space = {
        "gamma": np.arange(1, 20),
        "num_parallel_tree": np.linspace(10, 100, 10, dtype=int),
        "max_depth": np.arange(1, 25),
        "learning_rate": np.linspace(0.5, 0.01, 10),
        "subsample": np.linspace(1, 0.1, 20),
        "colsample_bynode": np.linspace(1, 0.1, 10),
        "n_estimators": np.linspace(10, 200, 20, dtype=int),
    }
    clf = RandomizedSearchCV(
        model, hp_space, scoring="neg_log_loss", verbose=2, random_state=0, n_jobs=-1
    )
    clf.fit(x, y, sample_weight=classes_weights)

    return clf.best_estimator_

def majority_vote(predictions, rule="hard"):
    """
    Performs majority voting on a list of predictions.

    Args:
        predictions: A list of lists, where each inner list contains predictions (classes) from one model.
        rule: "hard" for hard voting, "soft" for soft voting (default).

    Returns:
        A list of size equal to the number of inner lists, where each element is the majority class for the corresponding prediction across all models.
    """
    majority_classes = []
    for i in range(len(predictions[0])):

        if rule == "hard":
            class_counts = Counter([prediction[i] for prediction in predictions])
            majority_class = class_counts.most_common(1)[0][0]
        elif rule == "soft":
            majority_class = [0] * len(predictions[0][0])

            for model_predictions in predictions:
                majority_class = [
                    a + b for a, b in zip(majority_class, model_predictions[i])
                ]

            if sum(majority_class) > 0:
                majority_class = [p / sum(majority_class) for p in majority_class]

        else:
            raise ValueError("Invalid rule. Choose 'hard' or 'soft'.")
        majority_classes.append(majority_class)

    return majority_classes


def get_index_and_proba(data):
    """
    Finds the index and value of the highest element in each sublist.

    Args:
        data: A list of lists, where each inner list contains numerical values.

    Returns:
        A tuple containing two lists:
            - indices: A list containing the index of the highest element in each sublist.
            - values: A list containing the corresponding highest elements.
    """
    indices = []
    values = []

    for _, sublist in enumerate(data):
        # Find the index of the maximum value
        max_index = sublist.index(max(sublist))
        # Append the index and corresponding value
        indices.append(max_index)
        values.append(max(sublist))

    return indices, values


def predict_xg(data):
    """
    Predictions of the model for certain data. Model is saved in output/models.pickle

    Args:
        data: A numpy array to predict on.
    Returns:
        A numpy array of class predictions
    """

    with open("models/XGBoost/XGBoost_model.pickle", "rb") as f:
        models = pickle.load(f)

    y_preds = []
    for model in models:
        y_pred = model.predict_proba(data)
        y_preds.append(y_pred)

    maj_preds = majority_vote(y_preds, rule="soft")
    indices, _ = get_index_and_proba(maj_preds)

    return np.array(indices)
