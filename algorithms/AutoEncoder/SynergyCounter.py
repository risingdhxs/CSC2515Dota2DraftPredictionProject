# coding=utf-8
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
from tqdm import tqdm
from scipy import sparse


def preprocess_input_data(data_X):
    """
    Using synergy matrix (Ws) and opposition matrix (Wo) between hero, we have a prediction value calculated by:
    z = Xr*Wo*Xb.transform() + Xr*Ws*Xr.transform() - Xb*Ws*Xb.transform()
    Taking the diagonal values only and apply the activation function, we get the predictions.

    To make this equations fit the Logistic regression, we need to pre-process the input.
    :param data_X: np array/matrix or sparse matrix
    :return: processed data_X. If data_X size is (N, M), the output will be (N, (m//2)^2 * 2)
    """

    def process_one_row(x_row):
        if sparse.issparse(x_row):
            is_sparse = True
            x_red = sparse.csr_matrix(x_row[:, :M // 2])
            x_blue = sparse.csr_matrix(x_row[:, M // 2:])
        else:
            is_sparse = False
            x_red = np.asmatrix(x_row[:, :M // 2])
            x_blue = np.asmatrix(x_row[:, M // 2:])

        # transpose of x_red
        x_red_T = x_red.T

        # transpose of x_blue
        x_blue_T = x_blue.T

        opposition = (x_red_T * x_blue)
        opposition = opposition.reshape((1, (M // 2) ** 2))
        synergy = (x_red_T * x_red).reshape((1, (M // 2) ** 2)) - (x_blue_T * x_blue).reshape((1, (M // 2) ** 2))

        if is_sparse:
            result_x_row = sparse.hstack([opposition, synergy])
        else:
            result_x_row = np.hstack([opposition, synergy])

        return result_x_row

    N, M = data_X.shape

    # use half of the cores
    num_cores = multiprocessing.cpu_count() // 2

    print("Starting parallel running...")
    result_X = Parallel(n_jobs=num_cores)(delayed(process_one_row)(data_X[i]) for i in tqdm(range(N)))
    print("Parallel running finished.")

    if sparse.issparse(data_X):
        result_X = sparse.vstack(result_X)
    else:
        result_X = np.asarray(np.vstack(result_X))

    return result_X
