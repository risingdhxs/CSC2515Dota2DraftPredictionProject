#!/usr/bin/env python -W ignore::DeprecationWarning
import numpy as np
from autoencoder_113 import autoencoder
from SynergyCounter import preprocess_input_data
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import sys
import pickle

def samplecut(m, n):
    mem = 15
    lim = (7e8)*(mem/15)
    if 2*m*n*n > lim:
        return int(m* (lim / (2*m*n*n)))
    else:
        return m

def encodingX(X, modelpath):
    print('Encoding Input Data for model {}'.format(modelpath))
    model_param = torch.load(modelpath)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = autoencoder(model_param['layer']).to(device)
    model.load_state_dict(model_param['model_state_dict'])

    m = X.shape[0]
    n = X.shape[1]
    batch_size = 8192

    Xae = torch.from_numpy(np.vstack((X[:, :int(n / 2)], X[:, int(n / 2):]))).to(device)
    dataloader = DataLoader(Xae, batch_size=batch_size, shuffle=False)
    encodeX = np.zeros((0, model_param['layer'][-1]))

    for data in dataloader:
        encode, _ = model(data)
        encodeX = np.vstack((encodeX, encode.cpu().data.numpy()))
    encodeX = np.hstack((encodeX[:m, :], encodeX[m:, :]))
    return encodeX

datapath = '../../data/all/all_IO_noleave_VH.npz'
dataset = np.load(datapath)
TrainX = np.asmatrix(dataset['TrainX'])[0, 0]
ValidX = np.asmatrix(dataset['ValidX'])[0, 0]
TestX = np.asmatrix(dataset['TestX'])[0, 0]

TrainY = np.asarray(np.asmatrix(dataset['TrainY'])[0, 0].todense())
ValidY = np.asarray(np.asmatrix(dataset['ValidY'])[0, 0].todense())
TestY = np.asarray(np.asmatrix(dataset['TestY'])[0, 0].todense())

print('Fitting raw LRSC Model, VH')
lrsc_raw = LogisticRegression(solver='newton-cg', max_iter=100)
print('Expanding raw Training data into LRSC format')
TrainX_sc = preprocess_input_data(TrainX)
lrsc_raw.fit(TrainX_sc, TrainY.ravel())

filetitle = 'LRSC_Model_VH_raw.p'
pickle.dump(lrsc_raw, open(filetitle, 'wb'))

score_train_lrsc_raw = lrsc_raw.score(TrainX_sc, TrainY.ravel())
del TrainX_sc

ValidX_sc = preprocess_input_data(ValidX)
score_valid_lrsc_raw = lrsc_raw.score(ValidX_sc, ValidY.ravel())
del ValidX_sc

TestX_sc = preprocess_input_data(TestX)
score_test_lrsc_raw = lrsc_raw.score(TestX_sc, TestY.ravel())
del TestX_sc

LRSC_VH_performance = {'raw': [score_train_lrsc_raw, score_valid_lrsc_raw, score_test_lrsc_raw]}

del lrsc_raw

print('raw LRSC Model accuracy: {:.2f}/{:.2f}/{:.2f}'.format(score_train_lrsc_raw*100,
                                                             score_valid_lrsc_raw*100,
                                                             score_test_lrsc_raw*100))

datapath = '../../data/all/all_IO_noleave_VH.npz'
dataset = np.load(datapath)
files_VH = [f for f in os.listdir('../AutoEncoder/AutoEncoderResults/') if f.endswith('VH.pt')]
for i in range(len(files_VH)):
    modelpath = '../AutoEncoder/AutoEncoderResults/'+files_VH[i]
    model_param = torch.load(modelpath)
    layer = model_param['layer']
    if len(layer) == 2:
        nodenum = '{}-{}'.format(layer[0], layer[1])
    elif len(layer) == 3:
        nodenum = '{}-{}-{}'.format(layer[0], layer[1], layer[2])
    elif len(layer) == 4:
        nodenum = '{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3])
    elif len(layer) == 6:
        nodenum = '{}-{}-{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3], layer[4], layer[5])

    print('VH Model: {}'.format(nodenum))

    TrainX = np.asarray(np.asmatrix(dataset['TrainX'])[0, 0].astype(np.float32).todense())
    TrainY = np.asarray(np.asmatrix(dataset['TrainY'])[0, 0].todense())
    m = samplecut(TrainX.shape[0], layer[-1])
    if m != TrainX.shape[0]:
        print('Cutting down sample number from {} to {} due to memory constraint'.format(TrainX.shape[0], m))
    TrainX = TrainX[:m, :]
    TrainY = TrainY[:m]
    encodeX_train = encodingX(TrainX, modelpath)
    del TrainX

    encodeX_train_sc = preprocess_input_data(encodeX_train)
    del encodeX_train

    lrsc_encode = LogisticRegression(solver='newton-cg', max_iter=100)
    lrsc_encode.fit(encodeX_train_sc, TrainY.ravel())

    filetitle = 'LRSC_Model_VH_' + nodenum + '.p'
    pickle.dump(lrsc_encode, open(filetitle, 'wb'))

    score_train_lrsc_encode = lrsc_encode.score(encodeX_train_sc, TrainY.ravel())
    del encodeX_train_sc

    ValidX = np.asarray(np.asmatrix(dataset['ValidX'])[0, 0].astype(np.float32).todense())
    ValidY = np.asarray(np.asmatrix(dataset['ValidY'])[0, 0].todense())
    encodeX_valid = encodingX(ValidX[:min(m, ValidX.shape[0]), :], modelpath)
    del ValidX
    encodeX_valid_sc = preprocess_input_data(encodeX_valid)
    del encodeX_valid
    score_valid_lrsc_encode = lrsc_encode.score(encodeX_valid_sc, ValidY.ravel()[:encodeX_valid_sc.shape[0]])
    del encodeX_valid_sc

    TestX = np.asarray(np.asmatrix(dataset['TestX'])[0, 0].astype(np.float32).todense())
    TestY = np.asarray(np.asmatrix(dataset['TestY'])[0, 0].todense())
    encodeX_test = encodingX(TestX[:min(m, TestX.shape[0]), :], modelpath)
    del TestX
    encodeX_test_sc = preprocess_input_data(encodeX_test)
    del encodeX_test
    score_test_lrsc_encode = lrsc_encode.score(encodeX_test_sc, TestY.ravel()[:encodeX_test_sc.shape[0]])
    del encodeX_test_sc

    del lrsc_encode

    LRSC_VH_performance[nodenum] = [score_train_lrsc_encode, score_valid_lrsc_encode, score_test_lrsc_encode]
    print('VH {} LRSC Model accuracy: {:.2f}/{:.2f}/{:.2f}'.format(nodenum, score_train_lrsc_encode * 100,
                                                                   score_valid_lrsc_encode * 100,
                                                                   score_test_lrsc_encode * 100))

filetitle = 'LRSC_VH_Performance.p'
pickle.dump(LRSC_VH_performance, open(filetitle, 'wb'))


# Separator between VH case and N case

datapath = '../../data/all/all_IO_noleave_N.npz'
dataset = np.load(datapath)
TrainX = np.asmatrix(dataset['TrainX'])[0, 0]
ValidX = np.asmatrix(dataset['ValidX'])[0, 0]
TestX = np.asmatrix(dataset['TestX'])[0, 0]

TrainY = np.asarray(np.asmatrix(dataset['TrainY'])[0, 0].todense())
ValidY = np.asarray(np.asmatrix(dataset['ValidY'])[0, 0].todense())
TestY = np.asarray(np.asmatrix(dataset['TestY'])[0, 0].todense())

print('Fitting raw LRSC Model, N')
lrsc_raw = LogisticRegression(solver='newton-cg', max_iter=100)
print('Expanding raw Training data into LRSC format')
TrainX_sc = preprocess_input_data(TrainX)
lrsc_raw.fit(TrainX_sc, TrainY.ravel())

filetitle = 'LRSC_Model_N_raw.p'
pickle.dump(lrsc_raw, open(filetitle, 'wb'))

score_train_lrsc_raw = lrsc_raw.score(TrainX_sc, TrainY.ravel())
del TrainX_sc

ValidX_sc = preprocess_input_data(ValidX)
score_valid_lrsc_raw = lrsc_raw.score(ValidX_sc, ValidY.ravel())
del ValidX_sc

TestX_sc = preprocess_input_data(TestX)
score_test_lrsc_raw = lrsc_raw.score(TestX_sc, TestY.ravel())
del TestX_sc

LRSC_N_performance = {'raw': [score_train_lrsc_raw, score_valid_lrsc_raw, score_test_lrsc_raw]}

del lrsc_raw

print('raw LRSC Model accuracy: {:.2f}/{:.2f}/{:.2f}'.format(score_train_lrsc_raw*100,
                                                             score_valid_lrsc_raw*100,
                                                             score_test_lrsc_raw*100))

datapath = '../../data/all/all_IO_noleave_N.npz'
dataset = np.load(datapath)
files_N = [f for f in os.listdir('../AutoEncoder/AutoEncoderResults/') if f.endswith('N.pt')]
for i in range(len(files_N)):
    modelpath = '../AutoEncoder/AutoEncoderResults/'+files_N[i]
    model_param = torch.load(modelpath)
    layer = model_param['layer']
    if len(layer) == 2:
        nodenum = '{}-{}'.format(layer[0], layer[1])
    elif len(layer) == 3:
        nodenum = '{}-{}-{}'.format(layer[0], layer[1], layer[2])
    elif len(layer) == 4:
        nodenum = '{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3])
    elif len(layer) == 6:
        nodenum = '{}-{}-{}-{}-{}-{}'.format(layer[0], layer[1], layer[2], layer[3], layer[4], layer[5])

    print('N Model: {}'.format(nodenum))

    TrainX = np.asarray(np.asmatrix(dataset['TrainX'])[0, 0].astype(np.float32).todense())
    TrainY = np.asarray(np.asmatrix(dataset['TrainY'])[0, 0].todense())
    m = samplecut(TrainX.shape[0], layer[-1])
    if m != TrainX.shape[0]:
        print('Cutting down sample number from {} to {} due to memory constraint'.format(TrainX.shape[0], m))
    TrainX = TrainX[:m, :]
    TrainY = TrainY[:m]
    encodeX_train = encodingX(TrainX, modelpath)
    del TrainX

    encodeX_train_sc = preprocess_input_data(encodeX_train)
    del encodeX_train

    lrsc_encode = LogisticRegression(solver='newton-cg')
    lrsc_encode.fit(encodeX_train_sc, TrainY.ravel())

    filetitle = 'LRSC_Model_N_' + nodenum + '.p'
    pickle.dump(lrsc_encode, open(filetitle, 'wb'))

    score_train_lrsc_encode = lrsc_encode.score(encodeX_train_sc, TrainY.ravel())
    del encodeX_train_sc

    ValidX = np.asarray(np.asmatrix(dataset['ValidX'])[0, 0].astype(np.float32).todense())
    ValidY = np.asarray(np.asmatrix(dataset['ValidY'])[0, 0].todense())
    encodeX_valid = encodingX(ValidX[:min(m, ValidX.shape[0]), :], modelpath)
    del ValidX
    encodeX_valid_sc = preprocess_input_data(encodeX_valid)
    del encodeX_valid
    score_valid_lrsc_encode = lrsc_encode.score(encodeX_valid_sc, ValidY.ravel()[:encodeX_valid_sc.shape[0]])
    del encodeX_valid_sc

    TestX = np.asarray(np.asmatrix(dataset['TestX'])[0, 0].astype(np.float32).todense())
    TestY = np.asarray(np.asmatrix(dataset['TestY'])[0, 0].todense())
    encodeX_test = encodingX(TestX[:min(m, TestX.shape[0]), :], modelpath)
    del TestX
    encodeX_test_sc = preprocess_input_data(encodeX_test)
    del encodeX_test
    score_test_lrsc_encode = lrsc_encode.score(encodeX_test_sc, TestY.ravel()[:encodeX_test_sc.shape[0]])
    del encodeX_test_sc

    del lrsc_encode

    LRSC_N_performance[nodenum] = [score_train_lrsc_encode, score_valid_lrsc_encode, score_test_lrsc_encode]
    print('N {} LRSC Model accuracy: {:.2f}/{:.2f}/{:.2f}'.format(nodenum, score_train_lrsc_encode * 100,
                                                                   score_valid_lrsc_encode * 100,
                                                                   score_test_lrsc_encode * 100))

filetitle = 'LRSC_N_Performance.p'
pickle.dump(LRSC_N_performance, open(filetitle, 'wb'))







