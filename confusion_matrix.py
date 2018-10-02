from sklearn.metrics import confusion_matrix

from knn_naive import *


def confusion_matrix_fashion(act_arr, pred_arr):
    import numpy as np
    print('act_arr', act_arr)
    print('pred_arr', pred_arr)
    cm1 = confusion_matrix(act_arr, pred_arr)
    print('Confusion Matrix : \n', cm1)
    num_classes = 10
    TP = np.diag(cm1)
    print('TruePositive: ', TP)
    FP = []
    for i in range(num_classes):
        FP.append(sum(cm1[:, i]) - cm1[i, i])
    print('FalsePositive: ', FP)
    FN = []
    for i in range(num_classes):
        FN.append(sum(cm1[i, :]) - cm1[i, i])
    print('FalseNegative', FN)
    TN = []
    for i in range(num_classes):
        temp = np.delete(cm1, i, 0)
        temp = np.delete(temp, i, 1)
        TN.append(sum(sum(temp)))
    print('TrueNegative', TN)

    overall_accuracy=  TP + TN /(TP+ TN + FP + FN)
    print('overall accuracy: ', overall_accuracy)
    precision = TP /(TP+TN)
    print ('precision', precision)
