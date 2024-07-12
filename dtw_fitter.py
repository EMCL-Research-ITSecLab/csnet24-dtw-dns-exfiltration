import numpy as np
from fitter import Fitter, get_common_distributions, get_distributions
import pandas as pd
from scipy import stats
from sklearn.datasets import load_diabetes
from utils import HEICLOUD_DATA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pylab

if __name__ == "__main__":
    x_arr = []
    y_arr = []

    data_types = ["cic"]  # "live", "test", "dnscapy", "tuns", "plain"
    data_types = data_types #+ HEICLOUD_DATA

    # load data
    for data_type in data_types:
        y_arr.append(np.load(f"dtw_data_npy/y_{data_type}_30s_packet_size.npy"))
        x_arr.append(np.load(f"dtw_data_npy/x_{data_type}_30s_packet_size.npy"))

    X = np.concatenate(x_arr)
    y = np.concatenate(y_arr)
    y = y.reshape(-1)

    y1 = y[np.where(y == 1)[0]]
    X1 = X[np.where(y == 1)[0]]
    X2 = X[np.where(y == 2)[0]]

    # Organize Data - from question
    X1 = X1.reshape(-1)
    X1 = X1[X1 > 0]

    X2 = X2.reshape(-1)
    X2 = X2[X2 > 0]
    # SR_y = pd.Series(X, name="y_ (Target Vector Distribution)")

    # # fitter
    # distributions_set = get_common_distributions()
    # distributions_set.extend(
    #     [
    #         # "arcsine",
    #         # "cosine",
    #         # "expon",
    #         # "weibull_max",
    #         # "weibull_min",
    #         # "dweibull",
    #         # "t",
    #         # "pareto",
    #         # "exponnorm",
    #         # "lognorm",
    #         "norm",
    #         # "exponweib",
    #         # "weibull_max",
    #         # "weibull_min",
    #         # "pareto",
    #         # "genextreme",
    #     ]
    # )
    # f = Fitter(SR_y, distributions=["cauchy"], xmin=0, xmax=8)
    # f.fit(progress=True)
    # f.summary()
    # pylab.xlabel("Entropy")
    # pylab.ylabel("Relative")
    # pylab.xlim(0,8)
    
        
    # fit = stats.norm.pdf(X, np.mean(X), np.std(X))  #this is a fitting indeed
    sns.histplot(X1, bins=30, kde=True, stat='probability')
    sns.histplot(X2, bins=30, kde=True, stat='probability')
    plt.xlabel('Height (Men)')
    plt.ylabel('Probability')
    plt.title("Distribution of Men's Height")
    plt.xticks(range(30,80,5))
    plt.show()