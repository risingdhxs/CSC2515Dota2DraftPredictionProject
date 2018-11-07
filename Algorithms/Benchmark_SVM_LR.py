import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn import svm
from sklearn.ensemble import BaggingClassifier


trainX=sparse.load_npz('trainsetInputVector_sparse.npz')
trainY = np.genfromtxt('trainsetResult.csv', delimiter='\n')

validX=sparse.load_npz('validsetInputVector_sparse.npz')
validY = np.genfromtxt('validsetResult.csv', delimiter='\n')

testX=sparse.load_npz('testsetInputVector_sparse.npz')
testY = np.genfromtxt('testsetResult.csv', delimiter='\n')


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


# bagging of SVM with linear kernel to speed up training
n_estimators = 20
clf = BaggingClassifier(svm.LinearSVC(), max_samples=1.0 / n_estimators, n_estimators=n_estimators,n_jobs=n_estimators)
clf.fit(trainX, trainY)
clf.score(validX,validY) # valid_score = 0.6235049863363736
clf.score(validX_sub,validY_sub) # valid_score = 0.6213


# bagging of SVM with linear kernel to speed up training
n_estimators = 20
clf = BaggingClassifier(svm.SVC(kernel='sigmoid',), max_samples=1.0 / n_estimators, n_estimators=n_estimators,n_jobs=n_estimators)
clf.fit(trainX, trainY)
valid_score = clf.score(validX,validY) # valid_score = 0.6235049863363736

# bagging of SVM with linear kernel to speed up training
n_estimators = 20
clf = BaggingClassifier(svm.SVC(kernel='poly',), max_samples=1.0 / n_estimators, n_estimators=n_estimators,n_jobs=n_estimators)
clf.fit(trainX, trainY)
valid_score = clf.score(validX_sub,validY_sub) # valid_score = 0.53057











# SVM with linear kernel with random sampled a subset

clf = svm.LinearSVC()
clf.fit(trainX_sub, trainY_sub)
valid_score = clf.score(validX_sub,validY_sub) # valid_score = 0.62153

# SVM with linear kernel with random sampled a subset with different C

res = []
for c in np.arange(1,100,10):
    print('C=%d' % c)
    clf = svm.LinearSVC(C=c)
    clf.fit(trainX_sub, trainY_sub)
    res.append(clf.score(validX_sub,validY_sub))
plt.plot(np.arange(1,100,10),res)
plt.ylabel('Accuracy')
plt.xlabel('Penalty the error term')
plt.savefig('svm_poly_diff_Cs.png')

# SVM with poly with random sampled a subset
n_estimators = 10
clf = BaggingClassifier(svm.SVC(kernel='poly',), max_samples=1.0 / n_estimators, n_estimators=n_estimators,n_jobs=n_estimators)
clf.score(validX_sub,validY_sub) # valid_score = 0.53018
clf = svm.SVC(kernel='poly',)
clf.fit(trainX_sub, trainY_sub) 
clf.score(validX_sub,validY_sub) # valid_score = 






# PCA
from sklearn.decomposition import PCA

pca = PCA()
trainX_transformed = pca.fit_transform(trainX.todense())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
trainX_normalized = scaler.fit_transform(trainX.todense())
trainX_transformed_normalized = scaler.fit_transform(trainX_transformed)

import seaborn as sns
for i in range(10):
    print('processing %d' % i)
    plt.clf()
    sns.distplot(trainX_transformed_normalized[trainY==0][:,i], rug=True) 
    plt.savefig('pcs/pc%d_0.png' % i)
    plt.clf()
    sns.distplot(trainX_transformed_normalized[trainY==1][:,i], rug=True)
    plt.savefig('pcs/pc%d_1.png' % i)






from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=0.1, solver='lbfgs', n_jobs=20)
logreg.fit(trainX, trainY)
logreg.score(validX_sub,validY_sub) # score = 0.62503
scores = []
for C in [1,10,1e3,1e4,1e5]:
    print('Processing %d' % C)
    logreg = LogisticRegression(C=C, solver='lbfgs', n_jobs=20)
    logreg.fit(trainX, trainY)
    scores.append(logreg.score(validX_sub,validY_sub))


logreg = LogisticRegression(C=1e5, solver='saga', penalty='l1', n_jobs=20)
logreg.fit(trainX, trainY)
logreg.score(validX_sub,validY_sub) # score = 0.62176




logreg = LogisticRegression(C=1e5, solver='lbfgs', n_jobs=10)
from sklearn.decomposition import PCA
pca = PCA()
trainX_transformed = pca.fit_transform(trainX.todense())
scaler = MinMaxScaler()
trainX_transformed_normalized = scaler.fit_transform(trainX_transformed)

logreg.fit(trainX_transformed_normalized, trainY)
pca = PCA()
validX_transformed = pca.fit_transform(validX.todense())
scaler = MinMaxScaler()
validX_transformed_normalized = scaler.fit_transform(validX_transformed)
logreg.score(validX_transformed_normalized,validY)
























