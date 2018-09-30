import numpy as np
import matplotlib.pyplot as plot
import h5py
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import time
import cProfile


from sklearn.metrics import accuracy_score
with h5py.File('./data/images_training.h5','r') as H:
    data = np.copy(H['data'])
with h5py.File('./data/labels_training.h5','r') as H:
    label = np.copy(H['label'])
with h5py.File('./data/images_testing.h5', 'r') as H:
    test = np.copy(H['data'])
with h5py.File('./data/labels_testing_2000.h5', 'r') as H:
    test_label = np.copy(H['label'])

    train_X = data.reshape(30000,784)
    test_X = test.reshape(5000, 784)
    test_X = test_X[:2000]

    pca = PCA(3)

    pca.fit(train_X)
    comp = pca.components_
    #np.save('components_vector.npy',comp)
    redMat = pca.transform(train_X)
    print('redmat is :', redMat.shape)
    redMatTest = pca.transform(test_X)
    print('redMatTest is :', redMatTest.shape)

    def kdtree():
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=5, p=1, weights='distance', algorithm='kd_tree')
        knn.fit(redMat, label)
        y_pred = knn.predict(redMatTest)
        print('with pca, accuracy', accuracy_score(test_label, y_pred), 'time ', (time.time() - start_time))


    cProfile.run('kdtree()')


    def knn_with_pca():
        knn = KNeighborsClassifier(n_neighbors=5, p=1, weights='distance', algorithm='brute')
        knn.fit(redMat, label)
        y_pred = knn.predict(redMatTest)
        print('with pca, accuracy is', accuracy_score(test_label, y_pred))


    cProfile.run('knn_with_pca()')

    def knn_no_pca():
        knn = KNeighborsClassifier(n_neighbors=5, p=1, weights='distance',algorithm='brute')
        knn.fit(train_X,label)
        y_pred = knn.predict(test_X)
        print('no pca', accuracy_score(test_label, y_pred))


    #cProfile.run('knn_no_pca()')

