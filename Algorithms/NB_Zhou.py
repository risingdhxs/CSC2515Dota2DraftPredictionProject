# coding=utf-8
from scipy import sparse
import os
import pandas as pd

"""
Gaussian NB validation set:     0.602545
Gaussian NB test set:           0.601770
Multinomial NB validation set:  0.622438
Multinomial NB test set:        0.621305
Bernoulli NB validation set:    0.622127
Bernoulli NB test set:          0.621249
"""


def load_data(dataX_path, dataY_path, data_root='data/processed'):
    dataX = sparse.load_npz(os.path.join(data_root, dataX_path))
    dataY = pd.read_csv(os.path.join(data_root, dataY_path), header=None)

    print('Input Matrix Shape {0} x {1}, Target Shape {2} x {3}.'.format(dataX.shape[0], dataX.shape[1],
                                                                         dataY.shape[0], dataY.shape[1]))
    return dataX.toarray(), dataY.values.ravel()


def start_naive_bayes():
    trainX, trainY = load_data('trainsetInputVector_sparse.npz', 'trainsetResult.csv')
    validX, validY = load_data('validsetInputVector_sparse.npz', 'validsetResult.csv')
    testX, testY = load_data('testsetInputVector_sparse.npz', 'testsetResult.csv')

    from sklearn.naive_bayes import GaussianNB
    gauss_nb = GaussianNB()
    gauss_nb.fit(trainX, trainY)
    gauss_score = gauss_nb.score(validX, validY)
    print("Validation accuracy using Gaussian NB: %f" % gauss_score)

    gauss_score = gauss_nb.score(testX, testY)
    print("Test set accuracy using Gaussian NB: %f" % gauss_score)

    from sklearn.naive_bayes import MultinomialNB
    multi_nb = MultinomialNB()
    multi_nb.fit(trainX, trainY)
    multi_score = multi_nb.score(validX, validY)
    print("Validation accuracy using Multinomial NB: %f" % multi_score)

    multi_score = multi_nb.score(testX, testY)
    print("Test set accuracy using Multinomial NB: %f" % multi_score)

    from sklearn.naive_bayes import BernoulliNB
    bern_nb = BernoulliNB()
    bern_nb.fit(trainX, trainY)
    bern_score = bern_nb.score(validX, validY)
    print("Validation accuracy using Bernoulli NB: %f" % bern_score)

    bern_score = bern_nb.score(testX, testY)
    print("Test set accuracy using Bernoulli NB: %f" % bern_score)


start_naive_bayes()