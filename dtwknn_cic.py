import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn import metrics

y = np.load("data/y_cic_1min_packet_count.npy")
X = np.load("data/x_cic_1min_packet_count.npy")

y2 = np.load("data/y_heicloud_1min_packet_count.npy")
X2 = np.load("data/x_heicloud_1min_packet_count.npy")

X = np.concatenate([X, X2])
y = np.concatenate([y, y2])

import matplotlib.pyplot as plt

ax = plt.axes()
ax.xaxis.grid(which="both")
ax.set_ylabel("Amplitude")
ax.set_xlabel("Time")
for idx, j in enumerate(X):
    if y[idx] == 1:
        c = "r"
    else:
        c = "b"

    plt.plot(j, linewidth=3, color=c)

plt.minorticks_on()
plt.savefig("dist_data.pdf")
plt.show()
plt.clf()


scaler = TimeSeriesScalerMeanVariance()  # Rescale time series
X = scaler.fit_transform(X)

# the length of the time-series
sz = X.shape[1]

path, sim = metrics.dtw_path(X[0], X[1])
print(sim)

plt.figure(1, figsize=(8, 8))

# definitions for the axes
left, bottom = 0.01, 0.1
w_ts = h_ts = 0.2
left_h = left + w_ts + 0.02
width = height = 0.65
bottom_h = bottom + height + 0.02

rect_s_y = [left, bottom, w_ts, height]
rect_gram = [left_h, bottom, width, height]
rect_s_x = [left_h, bottom_h, width, h_ts]

ax_gram = plt.axes(rect_gram)
ax_s_x = plt.axes(rect_s_x)
ax_s_y = plt.axes(rect_s_y)

mat = cdist(X[0], X[1])

ax_gram.imshow(mat)
ax_gram.axis("off")
ax_gram.autoscale(False)
ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-", linewidth=3.0)

ax_s_x.plot(np.arange(sz), X[1], "b-", linewidth=3.0)
ax_s_x.axis("off")
ax_s_x.set_xlim((0, sz - 1))

ax_s_y.plot(-X[0], np.arange(sz)[::-1], "r-", linewidth=3.0)
ax_s_y.axis("off")
ax_s_y.set_ylim((0, sz - 1))

plt.savefig("dtw_path.pdf")
plt.show()
plt.clf()

print(X.shape)
print(y.shape)

# Count indices that are greater 0
x_occ = np.argwhere(X > 0)
# Get counts
unique, counts = np.unique(x_occ[:, 0], return_counts=True)
data = dict(zip(unique, counts))
results = data.items()
filterdata = np.array(list(results))

X = X[np.where(filterdata[:, 1] > 1)]
y = y[np.where(filterdata[:, 1] > 1)]

print(np.where(y == 1)[0].shape)
print(np.where(y == 2)[0].shape)

y2 = y[np.where(y == 2)[0]]
X2 = X[np.where(y == 2)[0]]

y1 = y[np.where(y == 1)[0]][:1000]
X1 = X[np.where(y == 1)[0]][:1000]

X = np.concatenate([X1, X2])
y = np.concatenate([y1, y2])

print(X.shape)
print(y.shape)

y = y.reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

knn = KNeighborsTimeSeries(n_neighbors=len(X_train))
knn.fit(X_train)

q = X_test[1].reshape(1, X_test.shape[1], 1)
ret = knn.kneighbors(q)
nn = ret[1][0][0]  # The nearest neighbour
fn = ret[1][0][-1]  # The farthest neighbor
nn, fn

plt.plot(q[0], linewidth=2, color="r")
plt.plot(X_train[nn], linewidth=2, color="b")
plt.title("The Nearest Neighbour")
plt.savefig("near.pdf")
plt.show()
plt.clf()

plt.plot(q[0], linewidth=2, color="r")
plt.plot(X_train[fn], linewidth=2, color="g")
plt.title("The Farthest Neighbour")
plt.savefig("far.pdf")
plt.show()
plt.clf()

# train
parameters = {"n_neighbors": [2, 4, 8, 10, 12, 14, 16]}
clf = GridSearchCV(
    KNeighborsTimeSeriesClassifier(metric="dtw", n_jobs=-1),
    parameters,
    cv=2,
    n_jobs=-1,
    verbose=2,
)
clf.fit(X_train, y_train)
print(clf.best_params_["n_neighbors"])

# evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


sampl = np.random.uniform(low=120, high=130, size=(100,))
y_pred = clf.predict([sampl])
print(y_pred)
