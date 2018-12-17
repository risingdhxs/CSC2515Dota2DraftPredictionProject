import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

print('Loading data...')
datapath = '../../data/all/all_IO_noleave_VH.npz'
datasetVH = np.load(datapath)
datapath = '../../data/all/all_IO_noleave_N.npz'
datasetN = np.load(datapath)
datapath = '../../data/all/all_IO_noleave.npz'
datasetall = np.load(datapath)

TrainX_all = np.asmatrix(datasetall['TrainX'])[0, 0]
TrainY_all = np.asarray(np.asmatrix(datasetall['TrainY'])[0, 0].todense())
TrainX_N = np.asmatrix(datasetN['TrainX'])[0, 0]
TrainY_N = np.asarray(np.asmatrix(datasetN['TrainY'])[0, 0].todense())
TrainX_VH = np.asmatrix(datasetVH['TrainX'])[0, 0]
TrainY_VH = np.asarray(np.asmatrix(datasetVH['TrainY'])[0, 0].todense())
TestX_VH = np.asmatrix(datasetVH['TestX'])[0, 0]
TestX_N = np.asmatrix(datasetN['TestX'])[0, 0]

print('Fitting LR_all Model...')
lr = LogisticRegression(solver='newton-cg')
lr.fit(TrainX_all, TrainY_all.ravel())
print('Fitting LR_N Model...')
lr_N = LogisticRegression(solver='newton-cg')
lr_N.fit(TrainX_N, TrainY_N.ravel())
print('Fitting LR_VH Model...')
lr_VH = LogisticRegression(solver='newton-cg')
lr_VH.fit(TrainX_VH, TrainY_VH.ravel())

print('Plotting Histogram...')
TestYprob_VH_all = lr.predict_proba(TestX_VH)
TestYprob_N_all = lr.predict_proba(TestX_N)
TestYprob_VH_N = lr_N.predict_proba(TestX_VH)
TestYprob_N_N = lr_N.predict_proba(TestX_N)
TestYprob_VH_VH = lr_VH.predict_proba(TestX_VH)
TestYprob_N_VH = lr_VH.predict_proba(TestX_N)

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.hist(TestYprob_VH_all[:,1], bins = 20)
# ax1.grid(True)
# ax1.set_title('LR_all Model Prediction Distribution on VH Games\n mean={:2.2f}%, std={:2.2f}%'.format(np.mean(TestYprob_VH_all[:,1]), np.std(TestYprob_VH_all[:,1])))
# ax2.hist(TestYprob_N_all[:,1], bins = 20)
# ax2.grid(True)
# ax2.set_xlabel('Predicted Win Probability')
# ax2.set_ylabel('Game Count')
# ax2.set_title('Distribution on N Games: mean={:2.2f}%, std={:2.2f}%'.format(np.mean(TestYprob_N_all[:,1]), np.std(TestYprob_N_all[:,1])))
#
# fig.savefig('Difference in LR_all Prediction Distribution between N and VH Sets.jpg')


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.hist(TestYprob_VH_N[:,1], bins = 20)
ax1.grid(True)
ax1.set_title('LR_N Model Prediction Distribution on VH_test Games\n mean={:2.2f}%, std={:2.2f}%'.format(np.mean(TestYprob_VH_N[:,1]), np.std(TestYprob_VH_N[:,1])))
ax2.hist(TestYprob_N_N[:,1], bins = 20)
ax2.grid(True)
ax2.set_title('LR_N Prediction on N_test Games: mean={:2.2f}%, std={:2.2f}%'.format(np.mean(TestYprob_N_N[:,1]), np.std(TestYprob_N_N[:,1])))
ax2.set_xlabel('Predicted Win Probability')
ax2.set_ylabel('Game Count')

fig.savefig('Difference in LR_N Prediction Distribution between N and VH Sets.jpg')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.hist(TestYprob_VH_VH[:,1], bins = 20)
ax1.grid(True)
ax1.set_title('LR_VH Model Prediction Distribution on VH_test Games\n mean={:2.2f}%, std={:2.2f}%'.format(np.mean(TestYprob_VH_VH[:,1]), np.std(TestYprob_VH_VH[:,1])))
ax2.hist(TestYprob_N_VH[:,1], bins = 20)
ax2.grid(True)
ax2.set_title('LR_VH Prediction on N_test Games: mean={:2.2f}%, std={:2.2f}%'.format(np.mean(TestYprob_N_VH[:,1]), np.std(TestYprob_N_VH[:,1])))
ax2.set_xlabel('Predicted Win Probability')
ax2.set_ylabel('Game Count')

fig.savefig('Difference in LR_VH Prediction Distribution between N and VH Sets.jpg')


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.hist(TestYprob_N_VH[:,1], bins = 20)
ax1.grid(True)
ax1.set_title('LR_VH Model Prediction Distribution on N_test Games\n mean={:2.2f}%, std={:2.2f}%'.format(np.mean(TestYprob_N_VH[:,1]), np.std(TestYprob_N_VH[:,1])))
ax2.hist(TestYprob_N_N[:,1], bins = 20)
ax2.grid(True)
ax2.set_title('LR_N Prediction on N_test Games: mean={:2.2f}%, std={:2.2f}%'.format(np.mean(TestYprob_N_N[:,1]), np.std(TestYprob_N_N[:,1])))
ax2.set_xlabel('Predicted Win Probability')
ax2.set_ylabel('Game Count')

fig.savefig('Difference in LR_VH and LR_N Prediction Distribution on N_test Sets.jpg')