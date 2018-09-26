import os
import numpy as np
import matplotlib.pyplot as plt


def getimpaths(datapath):
    paths = []
    for dir in os.listdir(datapath):
        try:
            for filename in os.listdir(os.path.join(datapath, dir)):
                paths.append(os.path.join(datapath, dir, filename))
        except:
            pass

    return paths


# m, n = np.array(Image.open(impaths[0])).shape[0:2]
# print m,n
# X = np.mat([np.array(Image.open(impath)).flatten() for impath in impaths])
 # X.shape= (400,10304)
# print X

    """
    Initializing the Eigenfaces model.
    """


def pca(X):
    n, d = X.shape()
    mu = X.mean(axis=0)
    X = X - mu
    C = np.dot(X, X.T)
    [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    # print 'x.t shape', X.shape
    # print eigenvectors.shape
    eigenvectors = np.dot(X.T, eigenvectors)
    for i in range(n):
        eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    # or simply perform an economy size decomposition
    # eigenvectors , eigenvalues , variance = np.linalg.svd(X.T,
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    evalues_sum = sum(eigenvalues[:])  # include only the first k evectors/values so
    evalues_count = 0  # that they include approx. 85% of the energy
    evalues_energy = 0.0
    for evalue in eigenvalues:
        evalues_count += 1
        evalues_energy += evalue / evalues_sum
        if evalues_energy >= 0.80:
            break
    print('number of eigenfaces ',evalues_count)
    eigenvalues = eigenvalues[0:evalues_count].copy()
    eigenvectors = eigenvectors[:, 0:evalues_count].copy()
    weights = np.asanyarray(X) * eigenvectors
    # print np.asanyarray(X).shape
    # print eigenvectors.shape
    # print 'weights shape is', weights.shape
    return [weights.T, eigenvectors, mu]


def plot_eigenValue(X,m,n):
    V, EV, immean = pca(X)
    plt.gray()
    plt.subplot(2, 4, 1)
    plt.imshow(immean.reshape(m, n))
    for i in range(7):
        plt.subplot(2, 4, i + 2)
        plt.imshow(EV[:, i].reshape(m, n))
    plt.show()


def euclidean_distance(p, q):
    p = np.array(p).flatten()
    q = np.array(q).flatten()
    return np.sqrt(np.sum(np.power((p-q) ,2)))


def knn_classifier(img, m, n, X):
    print('start of face evaluation')
    w, EV, avg_facevector = pca(X)
    img_col = np.array(img, dtype='float64').flatten()
    img_col = np.reshape(img_col, (m*n, 1)).T
    img_col -= avg_facevector
    cur_weight = np.asarray(EV.T).dot(img_col.T)
    # print 'EVT', np.asarray(EV.T).shape
    # print 'img_col shape is ', img_col.shape
    diffs = w-cur_weight
    minErr = float('inf')
    for i in range(w.shape[1]):
        if euclidean_distance(w[:,i],cur_weight) < minErr:
            minErr = euclidean_distance(w[:,i],cur_weight)

    norms = np.linalg.norm(diffs, axis=0)
    closest_face_id = np.argmin(norms)

    return minErr,closest_face_id


# image.show()
