import numpy as np


def meanX(dataX):
    return np.mean(dataX,axis=0)

def pca(X, k):
    """ Apply principle component analysis to X

            Args:
                    X(numpy.array): an array represents test image
                    k(int): select K components to keep

            returns:
                    finalData: dimension reduced matrix
                    selectVec: transformation matrix to do dimension reduction
                    """
    average = meanX(X)
    m, n = np.shape(X)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = X - avgs
    covX = np.cov(data_adjust.T)   # caculate correlation variance
    eigenValue, eigenVector=  np.linalg.eig(covX)  # caculate eigenvalue and eigenvector of cor variance matrix
    index = np.argsort(-eigenValue) # sort eigenValue as descending order
    redMat = []
    if k > n:
        print ("k must lower than feature number")
        return
    else:
        selectVec = np.matrix(eigenVector.T[index[:k]]) #transposed eigenVector
        redMat = data_adjust * selectVec.T
    return redMat, selectVec


def transform(X):
    """
    Apply dimensionality reduction to X.
    X is projected on the first principal components previously extracted from a training set.

        Args:
                X(numpy.array): an array represents test image

    """
    selectVec = np.load('components_vector.npy')
    mean_x = np.mean(X, axis=0)
    centralized_x = X - mean_x
    transformed_x = np.dot(centralized_x, selectVec.T)
    return transformed_x


