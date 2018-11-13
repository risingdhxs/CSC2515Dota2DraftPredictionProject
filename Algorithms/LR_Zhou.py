# coding=utf-8
from scipy import sparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

"""
validation accuracy using newton-cg:    0.6235523062545102
validation accuracy using lbfgs:        0.6235477057069135
validation accuracy using liblinear:    0.623549020149084
validation accuracy using sag:          0.6235463912647431
validation accuracy using saga:         0.6235509918123397

test set accuracy using newton-cg: 0.622607879555035
"""


def load_data(dataX_path, dataY_path, data_root='data/processed'):
    dataX = sparse.load_npz(os.path.join(data_root, dataX_path))
    dataY = pd.read_csv(os.path.join(data_root, dataY_path), header=None)

    print('Input Matrix Shape {0} x {1}, Target Shape {2} x {3}.'.format(dataX.shape[0], dataX.shape[1],
                                                                         dataY.shape[0], dataY.shape[1]))
    return dataX, dataY


def start_lr():
    trainX, trainY = load_data('trainsetInputVector_sparse.npz', 'trainsetResult.csv')
    validX, validY = load_data('validsetInputVector_sparse.npz', 'validsetResult.csv')
    testX, testY = load_data('testsetInputVector_sparse.npz', 'testsetResult.csv')

    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    best_score = 0
    best_solver = 'liblinear'

    for sol in solvers:
        lr = LogisticRegression(random_state=0, solver=sol)
        lr.fit(trainX, trainY.values.ravel())
        score = lr.score(validX, validY)
        if score > best_score:
            best_score = score
            best_solver = sol
        print("Validation accuracy using {0}: {1}".format(sol, score))

    lr = LogisticRegression(random_state=0, solver=best_solver)
    lr.fit(trainX, trainY.values.ravel())
    test_score = lr.score(testX, testY)

    print("Test set accuracy using {0}: {1}".format(best_solver, test_score))


start_lr()
