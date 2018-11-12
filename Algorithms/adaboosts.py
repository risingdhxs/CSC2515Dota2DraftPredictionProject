import random
import numpy as np
import matplotlib.pyplot as plt
import time
# start_time = time.time()
# main()
# print("--- %s seconds ---" % (time.time() - start_time))

from scipy import sparse
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

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

start_time = time.time()
clf = svm.LinearSVC()
clf.fit(trainX_sub, trainY_sub)
clf.score(validX_sub,validY_sub)
print("--- %s seconds ---" % (time.time() - start_time))


clf = AdaBoostClassifier(base_estimator=svm.LinearSVC(), n_estimators=100, algorithm='SAMME')
start_time = time.time()
clf.fit(trainX_sub, trainY_sub)
clf.score(validX_sub,validY_sub)
print("--- %s seconds ---" % (time.time() - start_time))



class AdaboostWithSVM:
    def __init__(self, clf, time, score):
        self.clf = clf
        self.time = time
        self.score = score
    def getTime():
        return self.time
    def getScore():
        return self.score
    def getClassifier():
        return self.clf

res_lr = []
for n_estimators in np.arange(1,200,10):
    print('processing %d estimators' % n_estimators)
    clf = AdaBoostClassifier(base_estimator=LogisticRegression(C=0.1, solver='lbfgs', n_jobs=20), n_estimators=n_estimators, algorithm='SAMME')
    start_time = time.time()
    clf.fit(trainX, trainY)
    score = clf.score(validX,validY)
    print("used time --- %s seconds ---" % (time.time() - start_time))
    res_lr.append(AdaboostWithSVM(clf=clf, time=int(time.time() - start_time), score=score))


print([len(e.clf.estimators_) for e in res_lr])

print([e.score for e in res_lr])





































