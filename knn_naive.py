from collections import defaultdict
from numpy import *
import heapq
import h5py
import pca_evaluation
import confusion_matrix
"""KNN classifier implemented by brute-force and kd-tree

This module implement a KNN classifier

"""


class TreeNode(object):
    def __init__(self, coordinates,label):
        self.coordinates = coordinates
        self.label = label

    def set_label(self,label):
        self.label = label


class kNN(object):

    def __init__(self, k, algorithm='brute-force'):
        """initialize a  k-nearest neighbor classifier.

        Args:
            dataset(list): a list of pair (arr,label), while arr is n components
             of dataset and label is the category.
            k(int): the number of nearest neighbour
            algorithm(str): brute-force / k_dtree
            """

        self.train_set = []
        self.kd_tree = []
        self.k = k
        self.algorithm = algorithm
        self.dim = 0


    def manhattan_distance(self,A,B):
        """ manhattan distance

        Args:
            A(numpy.array) : img1
            B(numpy.array) : img2

        Returns manhattan distance of two images
        """

        dist = sum(absolute(A-B),axis=0)
        return dist

    def make_kd_tree(self, train_nodes, dim, index=0):
        """ construct_kd_tree

        Args:
            train_nodes(numpy.array): a 2D array (30000, 75) represent images
            dim(int): dimension of the point (75 in this case)
            index(int): index for traverse the tree

        Returns manhattan distance of two images
        """

        # Select axis based on depth so that axis cycles through all valid values
        train_nodes.sort(key=lambda x: x[index])  # Sort point list
        index = (index + 1) % dim
        # choose median as pivot element
        _median = len(train_nodes) >> 1

        # Create node and construct subtree recursively
        return (
            self.make_kd_tree(train_nodes[: _median], dim, index),
            self.make_kd_tree(train_nodes[_median + 1:], dim, index),train_nodes[_median])

    def get_knn_kdtree(self, kd_node, point, return_distances=True, i=0, heap=None):
        """ construct_kd_tree

                Args:
                    kd_node(TreeNode): the root of k_dtree
                    dim(int): dimension of the point (75 in this case)
                    point(numpy.array) : an array represents test image

                Returns manhattan distance of two images
                """

        is_root = not heap
        if is_root:
            heap = []  # construct a bounded priority queue.
        if kd_node:
            dist = self.manhattan_distance(point, kd_node[2])
            dx = kd_node[2][i] - point[i]
            if len(heap) < self.k:
                heapq.heappush(heap, (-dist, kd_node[2]))
            elif dist < -heap[0][0]:
                heapq.heappushpop(heap, (-dist, kd_node[2]))
            i = (i + 1) % self.dim
            # Goes into the left branch, and then the right branch if needed
            self.get_knn_kdtree(kd_node[dx < 0], point,return_distances, i, heap)
            if dx * dx < -heap[0][0]:  # -heap[0][0] is the largest distance in the heap
                self.get_knn_kdtree(kd_node[dx >= 0], point, return_distances, i, heap)
        if is_root:
            neighbors = sorted((-h[0], h[1]) for h in heap)
            return neighbors if return_distances else [n[1] for n in neighbors]

    def get_prediction(self, neighbours):
        """ get the majority of votes in k nearest neighbours

        Args:
            neighbours(list) : a list of category of K nearest neighbours

        Returns:
            majority of category of nearest neighbours
        """
        counter = defaultdict(int)
        for votes in neighbours:
            counter[votes] += 1  # collect votes of each neighbour

        majority = max(counter.values())  # get the majority votes
        # find the category of the majority votes
        for k, v in counter.items():
            if v == majority:
                return k

    def classify(self, point):
        """ get the majority of votes in k nearest neighbours

            Args:
                point(numpy.array) : an array represents test image

            Returns:
                prediction of category of a image
                """
        if self.algorithm == 'k_dtree':
            result =[]
            result.append(self.get_knn_kdtree(self.kd_tree, point,
                                              return_distances=True, i=0, heap=None))
            neighbours = []
            for node in result[0]:
                neighbours.append(node[0])
            return self.get_prediction(result)

        else:
            temp_imgs = self.train_set[:]  # a temp array to store potential image
            k_nearest_neighbors = []
            while len(k_nearest_neighbors) < self.k:
                # construct a distance matrix through brute-force
                distance_matrix = [self.manhattan_distance(x[0], point) for x in temp_imgs]
                # Find the nearest neighbor.
                best_distance = min(distance_matrix)
                index = distance_matrix.index(best_distance)
                k_nearest_neighbors.append(temp_imgs[index])

                # Remove the nearest neighbour from the temp image list.
                del temp_imgs[index]

            # get prediction through voting.
            prediction = self.get_prediction([value[1] for value in k_nearest_neighbors])
            return prediction

    def fit(self, redMat, label):
        """ fit the label and traning data to KNN classifier

            Args:
                redMat(numpy.array): trainning set
                label(numpy.array): label
                """
        if self.algorithm == 'brute-force':
            for i in range(len(redMat)):
                self.train_set.append((redMat[i, :], label[i]))
        elif self.algorithm == 'k_dtree':
            trian_set_node = []
            self.dim = redMat.shape[1]
            for i in range(len(redMat)):
                tn = TreeNode(redMat[i, :],label[i])  # construct a tree node
                trian_set_node.append(tn)   # fit tree node to knn classifier
                self.kd_tree = self.make_kd_tree(trian_set_node, self.dim)
        else:
            print('invalid arguments.')


def main():
    pred_array = []
    with h5py.File('./data/images_training.h5', 'r') as H:
        data = copy(H['data'])
    with h5py.File('./data/labels_training.h5', 'r') as H:
        label = copy(H['label'])
    with h5py.File('./data/images_testing.h5', 'r') as H:
        test = copy(H['data'])
    with h5py.File('./data/labels_testing_2000.h5', 'r') as H:
        test_label = copy(H['label'])
    train_X = data.reshape(30000, 784)
    test_X = test.reshape(5000, 784)
    test_X = test_X[:2000]
    redMatTest = pca_evaluation.transform(test_X)
    redMat = pca_evaluation.transform(train_X)
    classifier = kNN(5,'brute-force')
    classifier.fit(redMat, label)
    count = 0
    mycount =0
    for i in range(len(redMatTest)):
        if classifier.classify(redMatTest[i]) == test_label[i]:
            print('yes', test_label[i],' mycount: ', mycount)
            count += 1
            mycount += 1
            pred_array.append(classifier.classify(redMatTest[i]))
        else:
            print('no')
            mycount += 1
            pred_array.append(classifier.classify(redMatTest[i]))
    print(count)
    confusion_matrix.confusion_matrix_fashion(test_label, pred_array)


main()
