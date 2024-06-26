from collections import Counter

import queue
import threading
import joblib
import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import class_weight

QUEUE = queue.Queue()

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
    for (X, y) in zip(xs_test, ys_test):
        for model in models:
            dtest = xgb.QuantileDMatrix(X)
            probas = model.predict(dtest)
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
    # First, get hypertrain result. We assume data is similiar.
    best_params = hypertrain("cuda:0" if torch.cuda.is_available() else "cpu", xs_train[0], ys_train[0])
    # TODO Use optuna
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100, timeout=600)

    # print("Number of finished trials: ", len(study.trials))
    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: {}".format(trial.value))
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    i = 0
    if torch.cuda.is_available():
        for gpu in range(0, torch.cuda.device_count()):
            threading.Thread(target=train, args=(f"cuda:{gpu}", best_params,), daemon=True).start()
    else:
        threading.Thread(target=train, args=(f"cpu", best_params,), daemon=True).start()
    
    for X, y in zip(xs_train, ys_train):
        classes_weights = class_weight.compute_sample_weight(class_weight="balanced", y=y)
        dtrain = xgb.QuantileDMatrix(X, y, weight=classes_weights)
        QUEUE.put_nowait((i, dtrain))
        i += 1
    QUEUE.join()

def train(device_cuda, best_params):
    """
    Trains a model on the provided features (x) and labels (y).

    Args:
        model (sklearn classifier): Model that is trained.
        x (pandas.DataFrame or numpy.ndarray): The features used for training.
        y (pandas.Series or numpy.ndarray): The labels used for training.

    Returns:
        xgboost.XGBClassifier: The trained model.
    """
    
    while True:
        (i, dtrain) = QUEUE.get()
        
        params = dict()
        params["device"] = device_cuda
        params["tree_method"] = "hist"
        params["objective"] = "multi:softprob"
        params["num_class"] = "2"
        params.update(best_params)
        params.pop("n_estimators") # Remove from params
        
        model = xgb.train(params, dtrain, num_boost_round=best_params["n_estimators"],)
        
        joblib.dump(model, f"models/xgboost_model_{i}.pickle")
        
        QUEUE.task_done()


def hypertrain(device_cuda, x, y):
    """
    Trains a model on the provided features (x) and labels (y).

    Args:
        model (sklearn classifier): Model that is trained.
        x (pandas.DataFrame or numpy.ndarray): The features used for training.
        y (pandas.Series or numpy.ndarray): The labels used for training.

    Returns:
        xgboost.XGBClassifier: The trained model.
    """
    # TODO Use Optuna
    # (data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    # train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    # dtrain = xgb.DMatrix(train_x, label=train_y)
    # dvalid = xgb.DMatrix(valid_x, label=valid_y)

    # param = {
    #     "verbosity": 0,
    #     "objective": "binary:logistic",
    #     # use exact for small dataset.
    #     "tree_method": "exact",
    #     # defines booster, gblinear for linear functions.
    #     "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
    #     # L2 regularization weight.
    #     "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
    #     # L1 regularization weight.
    #     "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    #     # sampling ratio for training data.
    #     "subsample": trial.suggest_float("subsample", 0.2, 1.0),
    #     # sampling according to each tree.
    #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    # }

    # if param["booster"] in ["gbtree", "dart"]:
    #     # maximum depth of the tree, signifies complexity of the tree.
    #     param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
    #     # minimum child weight, larger the term more conservative the tree.
    #     param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
    #     param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
    #     # defines how selective algorithm is.
    #     param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
    #     param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    # if param["booster"] == "dart":
    #     param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
    #     param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
    #     param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
    #     param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # bst = xgb.train(param, dtrain)
    # preds = bst.predict(dvalid)
    # pred_labels = np.rint(preds)
    # accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    
    model = xgb.XGBClassifier(
        tree_method="hist", device=device_cuda, verbosity=0, silent=True
    )
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
    
    print(clf.best_params_)
    
    return clf.best_params_


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
