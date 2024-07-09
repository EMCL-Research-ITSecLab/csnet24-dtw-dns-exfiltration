import joblib
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dtw_utils import HEICLOUD_DATA


def fit(X, y, model_name="dtwknn"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # train
    parameters = {"n_neighbors": [2, 4, 8, 10, 12, 14, 16], "weights": ["uniform", "distance"]}
    clf = GridSearchCV(
        KNeighborsTimeSeriesClassifier(
            metric="dtw", n_jobs=-1 #, metric_params={"global_constraint": "itakura", "itakura_max_slope": 1.}
        ),
        parameters,
        cv=2,
        n_jobs=-1,
        verbose=2,
    )

    clf.fit(X_train, y_train)
    print(clf.best_params_)


    # evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, f"models/{model_name}.pickle")

    
def plot(X, y, clf):
    ''' 
    split data, fit, classify, plot and evaluate results 
    '''
    # init vars
    n_neighbors = clf.best_params_["n_neighbors"]
    weights = clf.best_params_["weights"]
    h           = .1  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold  = ListedColormap(['#FF0000', '#0000FF'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    print(f"Start predicting mesh {np.c_[xx.ravel(), yy.ravel()].shape}")
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points, x-axis = 'Glucose', y-axis = "BMI"
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)   
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("0/1 outcome classification (k = %i, weights = '%s')" % (n_neighbors, weights))
    plt.show()
    fig.savefig(weights +'_new.pdf')

if __name__ == "__main__":
    x_arr = []
    y_arr = []

    data_types = [ "cic", "live", "test"]  # "dnscapy", "tuns", "plain", 
    # data_types = data_types + HEICLOUD_DATA

    # load data
    for data_type in data_types:
        y_arr.append(np.load(f"dtw_data_npy/y_{data_type}_1min_entropy.npy"))
        x_arr.append(np.load(f"dtw_data_npy/x_{data_type}_1min_entropy.npy"))

    X = np.concatenate(x_arr)
    y = np.concatenate(y_arr)

    # Count indices that are greater 0
    x_occ = np.argwhere(X > 0)
    # Get counts
    unique, counts = np.unique(x_occ[:, 0], return_counts=True)
    data = dict(zip(unique, counts))
    results = data.items()
    filterdata = np.array(list(results))

    X = X[np.where(filterdata[:, 1] > 2)]
    y = y[np.where(filterdata[:, 1] > 2)]

    y2 = y[np.where(y == 2)[0]]#[:1000]
    X2 = X[np.where(y == 2)[0]]#[:1000]

    y1 = y[np.where(y == 1)[0]]#[:1000]
    X1 = X[np.where(y == 1)[0]]#[:1000]

    X = np.concatenate([X1, X2])
    y = np.concatenate([y1, y2])

    # X, indices = np.unique(X.round(decimals=4), return_index=True, axis=0)
    # y = y[indices]
    y = y.reshape(-1)
    
    print(X.shape)

    fit(X, y, model_name="dtwknn_test_2")
    
    clf = joblib.load("models/dtwknn_test_2.pickle")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    print(X_train[:,0].shape)
    
    plot(X, y, clf)
