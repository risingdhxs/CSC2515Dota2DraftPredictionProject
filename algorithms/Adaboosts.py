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


clf = AdaBoostClassifier(base_estimator=LogisticRegression(C=10, solver='lbfgs', n_jobs=20), n_estimators=1, algorithm='SAMME')
clf.fit(trainX, trainY)
score = clf.score(validX_sub,validY_sub)



clf=LogisticRegression(C=10, solver='lbfgs', n_jobs=20)
clf.fit(trainX, trainY)
score = clf.score(validX_sub,validY_sub)


import math



trainY[trainY==0] = -1
validY[validY==0] = -1
testY[testY==0] = -1

from sklearn.linear_model import LogisticRegression
def compute_acc(clfs,coefs):
    outputs=[]
    for i in range(len(clfs)):
        outputs.append(clfs[i].predict(validX)*coef[i])
    ans = np.sum(outputs,axis=0)
    ans[ans>=0] = 1
    ans[ans<0] = -1
    acc = sum((ans*validY)==1)/len(ans)
    return acc


def adaboost(trainX, trainY, n_iterations):
    # initialize sample weights
    sample_weights = np.array([1.] * trainX.shape[0])
    # initialize classifier
    clf=LogisticRegression(C=10, solver='lbfgs', n_jobs=20)
    coef = []
    clfs = []
    res = []
    for i in range(n_iterations):
        clf=LogisticRegression(C=10, solver='lbfgs', n_jobs=20)
        clf.fit(trainX, trainY, sample_weight=sample_weights)
        pre = clf.predict(trainX)
        err = np.sum(sample_weights[trainY!=pre])/np.sum(sample_weights)
        print('error rate of %dth iteration is: %f' % (i, err))
        # compute the alpha and store it
        alpha = 1/2*math.log((1-err)/err)
        print('alpha of %dth iteration is: %f' % (i, alpha))
        updated_weights = sample_weights[trainY!=pre] * math.exp(2*alpha)
        sample_weights[trainY!=pre] = updated_weights
        # updated_weights_right = sample_weights[trainY==pre] * math.exp(-1*alpha)
        # updated_weights_wrong = sample_weights[trainY!=pre] * math.exp(alpha)
        # sample_weights[trainY==pre] = updated_weights_right
        # sample_weights[trainY!=pre] = updated_weights_wrong
        clfs.append(clf)
        coef.append(alpha)
        res.append(compute_acc(clfs,coef))
    compute_acc(clfs,coef)




from sklearn.naive_bayes import BernoulliNB
def adaboost(trainX, trainY, n_iterations):
    # initialize sample weights
    sample_weights = np.array([1.] * trainX.shape[0])
    # initialize classifier
    clf=BernoulliNB()
    coef = []
    clfs = []
    pres = []
    for i in range(n_iterations):
        clf.fit(trainX, trainY, sample_weight=sample_weights)
        pre = clf.predict(trainX)
        err = np.sum(sample_weights[trainY!=pre])/np.sum(sample_weights)
        print('error rate of %dth iteration is: %f' % (i, err))
        # compute the alpha and store it
        alpha = 1/2*math.log((1-err)/err)
        print('alpha of %dth iteration is: %f' % (i, alpha))
        updated_weights = sample_weights[trainY!=pre] * math.exp(2*alpha)
        sample_weights[trainY!=pre] = updated_weights
        # updated_weights_right = sample_weights[trainY==pre] * math.exp(-1*alpha)
        # updated_weights_wrong = sample_weights[trainY!=pre] * math.exp(alpha)
        # sample_weights[trainY==pre] = updated_weights_right
        # sample_weights[trainY!=pre] = updated_weights_wrong
        clfs.append(clf)
        coef.append(alpha)
    output = []
    for i in range(n_iterations):
        output.append(clfs[i].predict(validX)*coef[i])
    ans = np.sum(output,axis=0)
    ans[ans>=0] = 1
    ans[ans<0] = -1
    acc = sum((ans*validY)==1)/len(ans)


































