import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from fastdtw import fastdtw
from sklearn import preprocessing
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn import metrics


y = np.load("y_data_h_dest.npy")
X = np.load("x_data_h_dest.npy")

print(X)
# X = np.delete(X, (26), axis=0)
# X = np.delete(X, (26), axis=0)

# print(np.argmax(np.max(X, axis=0)))
print(X.shape)

# fig = plt.scatter(np.arange(36199), X[:, 1], marker="o", c=y, s=25, edgecolor="k")

plt.plot(np.arange(1, 29), X.transpose())
plt.savefig("x.png")
