from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
from sklearn import preprocessing
import pandas as pd

y = np.load("y_data.npy")
X = np.load("x_data.npy")

X = preprocessing.normalize(X)
print(X)
# Count indices that are greater 0
x_occ = np.argwhere(X > 0)
print(x_occ)
print(x_occ[:, 0])
# Get counts
unique, counts = np.unique(x_occ[:, 0], return_counts=True)
data = dict(zip(unique, counts))
print(data.get(0))
# df = pd.DataFrame(data, index=[1])
# print(df)

results = data.items()
print(list(results)[:5])
filterdata = np.array(list(results))
print(filterdata)

X = X[np.where(filterdata[:, 1] > 1200)]
y = y[np.where(filterdata[:, 1] > 1200)]

y2 = y[np.where(y == 2)[0]]
X2 = X[np.where(y == 2)[0]]

y1 = y[np.where(y == 1)[0]]
X1 = X[np.where(y == 1)[0]]

# path = dtw.warping_path(X2[0], X1[0])
# dtwvis.plot_warping(X2[0], X1[0], path)
# distance = dtw.distance(X2[0], X1[0])

print(X1[0])
print(X2[0])


d, paths = dtw.warping_paths(X1[50], X2[0], window=1500, use_pruning=True)
best_path = dtw.best_path(paths)
fig, ax = dtwvis.plot_warpingpaths(X1[50], X2[0], paths, best_path)

fig.savefig("test.pdf")
