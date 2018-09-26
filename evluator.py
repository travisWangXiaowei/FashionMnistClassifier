import numpy as np
import matplotlib.pyplot as plot
import h5py
from sklearn.decomposition import PCA

with h5py.File('./data/images_training.h5','r') as H:
    data = np.copy(H['data'])
with h5py.File('./data/labels_training.h5','r') as H:
    label = np.copy(H['label'])

    train_X = data.reshape(30000,784)
    variance = np.var(train_X, axis=0) > 1000
    train_X = train_X[:, variance]
    print(train_X.shape)
    pca = PCA()
    pca.fit(train_X)
    explained_variance = pca.explained_variance_ratio_
    # ##Calculate cumulative explained ration
    cum_explained_variance = [np.sum(explained_variance[:i + 1]) for i in range(20, 201, 10)]
    X_axis = [i for i in range(20, 201, 10)]

    fig = plot.figure()
    plot.plot(X_axis, cum_explained_variance, 'g^')
    plot.yticks(cum_explained_variance)
    plot.xticks(X_axis)
    plot.ylabel("Explained Variance Ratio")
    plot.xlabel("No. of Components")
    plot.show()