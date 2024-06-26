import numpy as np
from fitter import Fitter, get_common_distributions, get_distributions
from sklearn.datasets import load_diabetes
from dtw_utils import HEICLOUD_DATA

if __name__ == "__main__":
    x_arr = []
    y_arr = []

    data_types = ["cic", "dnscapy", "tuns", "plain"]  # "live", "test"
    data_types = data_types + HEICLOUD_DATA[0]

    # load data
    for data_type in data_types:
        y_arr.append(np.load(f"data/y_{data_type}_1min_entropy.npy"))
        x_arr.append(np.load(f"data/x_{data_type}_1min_entropy.npy"))

    X = np.concatenate(x_arr)
    y = np.concatenate(y_arr)
    y = y.reshape(-1)

    y1 = y[np.where(y == 2)[0]]
    X1 = X[np.where(y == 2)[0]]

    # Organize Data - from question
    X = X1.reshape(-1)
    X = X[X > 0]
    SR_y = pd.Series(X, name="y_ (Target Vector Distribution)")

    # fitter
    distributions_set = get_common_distributions()
    distributions_set.extend(
        [
            # "arcsine",
            # "cosine",
            # "expon",
            # "weibull_max",
            # "weibull_min",
            # "dweibull",
            # "t",
            # "pareto",
            # "exponnorm",
            # "lognorm",
            "norm",
            # "exponweib",
            # "weibull_max",
            # "weibull_min",
            # "pareto",
            # "genextreme",
        ]
    )
    f = Fitter(SR_y, distributions=distributions_set)
    f.fit()
    f.summary()
    print(get_distributions())
