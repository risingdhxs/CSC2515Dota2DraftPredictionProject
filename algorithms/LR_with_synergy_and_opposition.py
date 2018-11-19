# coding=utf-8
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
import pickle
from tqdm import tqdm


def load_data(name):
    """
    TrainInputSparse
    ValidInputSparse
    TestInputSparse
    TrainTargetSparse
    ValidTargetSparse
    TestTargetSparse
    :return:
    """
    all_data = np.load('../data/AllSetSparseInOut_noleave_VH.npz')
    return np.asmatrix(all_data[name])[0, 0]


def preprocess_input_data(data_X):
    """
    Using synergy matrix (Ws) and opposition matrix (Wo) between hero, we have a prediction value calculated by:
    z = Xr*Wo*Xb.transform() + Xr*Ws*Xr.transform() - Xb*Ws*Xb.transform()
    Taking the diagonal values only and apply the activation function, we get the predictions.

    To make this equations fit the Logistic regression, we need to pre-process the input.
    :return:
    """

    def process_one_row(x_row):
        x_red = sparse.csr_matrix(x_row[:, :M // 2])  # convert to sparse matrix
        x_blue = sparse.csr_matrix(x_row[:, M // 2:])

        # transpose of x_red
        x_red_T = x_red.T

        # transpose of x_blue
        x_blue_T = x_blue.T

        opposition = (x_red_T * x_blue)
        # print("N: {0}, M: {1}".format(N, M))
        # print("Row shape {0} {1}".format(x_row.shape, type(x_row)))
        # print("Red T shape {0} {1}".format(x_red_T.shape, type(x_red_T)))
        # print("Red shape {0} {1}".format(x_red.shape, type(x_red)))
        # print("Blue T shape {0} {1}".format(x_blue_T.shape, type(x_blue_T)))
        # print("Blue shape {0} {1}".format(x_blue.shape, type(x_blue)))
        #
        # print("opposition size {0} {1}".format(opposition.shape, type(opposition)))
        # print((M // 2) ** 2)
        opposition = opposition.reshape((1, (M // 2) ** 2))
        synergy = (x_red_T * x_red).reshape((1, (M // 2) ** 2)) - (x_blue_T * x_blue).reshape((1, (M // 2) ** 2))

        result_x_row = sparse.hstack([opposition, synergy])

        return result_x_row

    N, M = data_X.shape

    # result will be a N x (2*113^2) matrix
    # this will cause memory error immediately
    # result_X = np.zeros(shape=(N, 2*(M//2)**2))
    # result_X = sparse.csr_matrix((N, 2*(M//2)**2))

    num_cores = multiprocessing.cpu_count() - 5

    print("Starting parallel running...")
    result_X = Parallel(n_jobs=num_cores)(delayed(process_one_row)(data_X[i].todense()) for i in tqdm(range(N)))
    print("Parallel running finished.")

    result_X = sparse.vstack(result_X)

    return result_X


def start_lr():
    train_X = load_data('TrainX')
    # train_X = train_X[0:500000]
    train_X = preprocess_input_data(train_X)
    print("Finish processing training data. Save it to LRTrainingSparseIn_all_data_VH.npz.")
    sparse.save_npz('LRTrainingSparseIn_all_data_VH.npz', train_X)

    train_Y = load_data('TrainY')
    # train_Y = train_Y[0:500000]
    train_Y = train_Y.todense()

    test_X = load_data('TestX')
    # test_X = test_X[:10000]
    test_X = preprocess_input_data(test_X)
    print("Finish processing test data. Save it to LRTestSparseIn_all_data_VH.npz.")
    sparse.save_npz('LRTestSparseIn_all_data_VH.npz', test_X)

    test_Y = load_data('TestY')
    # test_Y = test_Y[:10000]
    test_Y = test_Y.todense()

    print("Size of Train_X: {}".format(train_X.shape))
    print("Size of Train_Y: {}".format(train_Y.shape))
    print("Size of Test_X: {}".format(test_X.shape))
    print("Size of Test_X: {}".format(test_Y.shape))
    lr = LogisticRegression(random_state=0, solver='newton-cg')
    lr.fit(train_X, train_Y)
    test_score = lr.score(test_X, test_Y)

    print("Test set accuracy: {0}".format(test_score))

    # save model
    pickle.dump(lr, open('LR_with_synergy_opposition_No_Leaver_VH.m', 'wb'))


start_lr()

# use opposition only
# Size of Train_X: (500000, 12769)
# Size of Train_Y: (500000,)
# Size of Test_X: (40000, 12769)
# Size of Test_X: (40000,)
# Test set accuracy: 0.623125

# use synergy only
# Size of Train_X: (500000, 12769)
# Size of Train_Y: (500000,)
# Size of Test_X: (40000, 12769)
# Size of Test_X: (40000,)
# Test set accuracy: 0.62355


# use both matrix, trained on all data.
# All data train set accuracy: 0.6510093447072126
# All data test set accuracy: 0.641256527848429
# Normal games test set accuracy: 0.6631410126602785
# Very High Level games test set accuracy: 0.6228336155485257

# use both matrix, trained on VH matches
# Train set accuracy: 0.6532283457679894
# Test set accuracy: 0.609650446950107


# use both matrix, trained on N matches
# Train set accuracy:0.6731356031687561
# Test set accuracy:0.658794560049248


