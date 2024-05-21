from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
from sklearn import preprocessing
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import pandas as pd

y = np.load("data/y_cic_h_packet.npy")
X = np.load("data/x_cic_h_packet.npy")

scaler = TimeSeriesScalerMeanVariance()  # Rescale time series
X = scaler.fit_transform(X)

# Count indices that are greater 0
x_occ = np.argwhere(X > 0)

# Get counts
unique, counts = np.unique(x_occ[:, 0], return_counts=True)
data = dict(zip(unique, counts))
results = data.items()

filterdata = np.array(list(results))

X = X[np.where(filterdata[:, 1] > 1)]
y = y[np.where(filterdata[:, 1] > 1)]

y2 = y[np.where(y == 2)[0]]
X2 = X[np.where(y == 2)[0]]

y1 = y[np.where(y == 1)[0]]
X1 = X[np.where(y == 1)[0]]

print(X1[0])
print(X2[0])

d, paths = dtw.warping_paths(X1[1], X1[0], window=50, use_pruning=True)
best_path = dtw.best_path(paths)
fig, ax = dtwvis.plot_warpingpaths(X1[1], X1[0], paths, best_path)

fig.savefig("dtw-path.pdf")
