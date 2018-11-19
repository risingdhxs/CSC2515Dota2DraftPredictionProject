import random

import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn import svm

from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from scipy import sparse
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
trainX = sparse.load_npz('./trainsetInputVector_sparse.npz')
trainY = np.genfromtxt('./trainsetResult.csv', delimiter='\n')
validX = sparse.load_npz('./validsetInputVector_sparse.npz')
validY = np.genfromtxt('./validsetResult.csv', delimiter='\n')
testX = sparse.load_npz('./testsetInputVector_sparse.npz')
testY = np.genfromtxt('./testsetResult.csv', delimiter='\n')





n_samples = 100000
indices = random.sample(range(trainX.shape[0]), n_samples)
trainX_sub = trainX[indices]
trainY_sub = trainY[indices]

n_samples = 100000
indices = random.sample(range(validX.shape[0]), n_samples)
validX_sub = validX[indices]
validY_sub = validY[indices]

n_samples = 100000
indices = random.sample(range(testX.shape[0]), n_samples)
testX_sub = testX[indices]
testY_sub = testY[indices]




def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((2, train_data.shape[1]))
    # Compute means
    for i in range(means.shape[0]): # loop for each class
        means[i] = np.mean(train_data[train_labels==i],axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((2, train_data.shape[1], train_data.shape[1]))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for i in range(covariances.shape[0]): # loop for each class
        trainX = train_data[train_labels==i]
        expe = trainX - means[i]
        covariances[i] = 1 / trainX.shape[0] * np.matmul(expe.T, expe) + 0.01*np.identity(100)
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    classlikelihood = np.zeros((digits.shape[0],2))
    n = digits.shape[0]
    d = digits.shape[1]
    for i in range(n):
        for j in range(2):
            classlikelihood[i,j] = -d/2 * np.log(2*np.pi) - 1/2*np.log(np.linalg.det(covariances[j])) - 1/2*np.matmul(
                np.matmul(
                    digits[i]-means[j],np.linalg.inv(covariances[j])
                    )
                ,(digits[i]-means[j]).T
                    )
        print('class likelihood for the %dth data is %s' % (i,classlikelihood[i]))
    return classlikelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    classlikelihood = generative_likelihood(digits, means, covariances)
    posterior = 1/2*classlikelihood
    return posterior








from joblib import Parallel, delayed 
# A function that can be called to do work:
def work(arg):    
    valid_data_st, valid_data_end = arg    
    # cond_likelihood = conditional_likelihood(np.reshape(valid_data[valid_data_ind],(1,-1)), means, covariances, valid_data_ind)
    cond_likelihood = conditional_likelihood(train_data[valid_data_st:valid_data_end], means, covariances, valid_data_st)
    res = [np.argmax(cond_likelihood[i]) for i in range(cond_likelihood.shape[0])]
    return sum(res==train_labels[valid_data_st:valid_data_end])/len(res)





pca = PCA()
trainX_transformed = pca.fit_transform(trainX.todense())
scaler = MinMaxScaler()
trainX_transformed_normalized = scaler.fit_transform(trainX_transformed)
train_data = trainX_transformed_normalized[:,:100]
train_labels = trainY
means = compute_mean_mles(train_data, train_labels)
covariances = compute_sigma_mles(train_data, train_labels)


pca = PCA()
validX_transformed = pca.fit_transform(validX_sub.todense())
scaler = MinMaxScaler()
validX_transformed_normalized = scaler.fit_transform(validX_transformed)
valid_data = validX_transformed_normalized[:2000,:100]
valid_labels = validY_sub

arg = []
for i in np.arange(0,2000,100):
    arg.append((i,i+100))

results = Parallel(n_jobs=25, verbose=1, backend="threading")(map(delayed(work), arg))






res_train = classify_data(train_data, means, covariances)
train_acc = sum(res_train==train_labels)/len(res_train) 














