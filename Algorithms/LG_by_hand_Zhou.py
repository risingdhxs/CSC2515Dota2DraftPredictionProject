# coding=utf-8
from scipy import sparse
import os
import numpy as np
import pandas as pd

"""
Good reference website: https://blog.csdn.net/namelessml/article/details/52431266

Trained on 20K matches, error rate 0.46912441063699184.
Trained on whole training set, error rate 0.46912441063699184.
"""


def load_data(dataX_path, dataY_path, data_root='data/processed'):
    dataX = sparse.load_npz(os.path.join(data_root, dataX_path))
    dataY = pd.read_csv(os.path.join(data_root, dataY_path), header=None)

    print('Input Matrix Shape {0} x {1}, Target Shape {2} x {3}.'.format(dataX.shape[0], dataX.shape[1],
                                                                         dataY.shape[0], dataY.shape[1]))
    return dataX, dataY


def sigmoid(in_x):
    """
    Activation function.
    :param in_x: a vector.
    :return: a vector.
    """
    # scipy sprase matrix does not implement np.exp
    return 1.0/(1+np.exp(-in_x))


def gradient_descent(trainX, trainY):
    # add constant 1 as the coefficient of b
    # use hstack to join sparse matrix
    trainX = sparse.hstack((np.ones((trainX.shape[0], 1)), trainX))

    # m denotes the number of training samples, n denotes the number of features
    m, n = trainX.shape

    alpha = 0.001  # learning rate
    max_iters = 1000  # maximum iterations allowed
    iter_cnt = 0  # iteration count

    # initial weights to 1
    weights = np.ones((n, 1))

    while iter_cnt < max_iters:
        z = trainX * weights  # m x 1 vector
        y = sigmoid(z)  # prediction after activation

        # calculate error
        error = trainY - y

        weights = weights - alpha * trainX.transpose() * error
        iter_cnt += 1
        print("iteration {} finished.".format(iter_cnt))
    return weights


def classify_prediction(in_x):
    """
    Classify y to 0 or 1.
    :param in_x: a vector.
    :return: a vector.
    """
    return np.where(in_x > 0.5, 1.0, 0.0)


def evaluate_error(dataX, dataY, weights):
    dataX = sparse.hstack((np.ones((dataX.shape[0], 1)), dataX))
    y = classify_prediction(sigmoid(dataX * weights))
    fit_cnt = ((y == dataY).sum()).values[0]
    return 1 - fit_cnt / (1.0 * dataY.shape[0])


def start_lg():
    trainX, trainY = load_data('trainsetInputVector_sparse.npz', 'trainsetResult.csv')
    print('Training started')
    weights = gradient_descent(trainX, trainY)
    print('Weights trained')
    testX, testY = load_data('testsetInputVector_sparse.npz', 'testsetResult.csv')
    print(evaluate_error(testX, testY, weights))


start_lg()
