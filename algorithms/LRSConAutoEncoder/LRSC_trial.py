import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from autoencoder_113 import autoencoder
from SynergyCounter import preprocess_input_data
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.data import DataLoader

def samplecut(m, n):
    mem = 15
    lim = (8e8)*(mem/15)
    # print('m={}, n={}, 2*m*n*n={}'.format(m,n,2*m*n*n))
    # print('cut ratio is {}'.format((lim / (2*m*n*n))))
    if 2*m*n*n > lim:
        return int(m* (lim / (2*m*n*n)))
    else:
        return m

def encodingX(X, modelpath):
    print('Encoding Input Data for model {}'.format(modelpath))
    model_param = torch.load(modelpath, map_location=lambda storage, loc: storage)
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

# print('Fitting raw LRSC Model')
# lrsc_raw = LogisticRegression(solver='newton-cg', max_iter=1000)
# print('Expanding raw Training data into LRSC format')
# TrainX_sc = preprocess_input_data(TrainX)
# lrsc_raw.fit(TrainX_sc, TrainY.ravel())
# score_train_lrsc_raw = lrsc_raw.score(TrainX_sc, TrainY.ravel())
# del TrainX_sc
# print('raw LRSC Model fit accuracy: {:.2f}'.format(score_train_lrsc_raw*100))
#
# ValidX_sc = preprocess_input_data(ValidX)
# score_valid_lrsc_raw = lrsc_raw.score(ValidX_sc, ValidY.ravel())
# del ValidX_sc
#
# TestX_sc = preprocess_input_data(TestX)
# score_test_lrsc_raw = lrsc_raw.score(TestX_sc, TestY.ravel())
# del TestX_sc
#
# print('raw LRSC Model accuracy: {:.2f}/{:.2f}/{:.2f}'.format(score_train_lrsc_raw*100,
#                                                              score_valid_lrsc_raw*100,
#                                                              score_test_lrsc_raw*100))
#
# del lrsc_raw

modelpath = '../AutoEncoder/AutoEncoderResults/Weighted_ReLu_AutoEncoder_100-75-50_all_IO_noleave_N.pt'
model_param = torch.load(modelpath, map_location=lambda storage, loc: storage)
print('Model: {}'.format(modelpath))
print('Encoding Input data using pretrained AutoEncoder')

TrainX = np.asarray(np.asmatrix(dataset['TrainX'])[0, 0].astype(np.float32).todense())
TrainY = np.asarray(np.asmatrix(dataset['TrainY'])[0, 0].todense())
m = samplecut(TrainX.shape[0], model_param['layer'][-1])
if m != TrainX.shape[0]:
    print('Cutting down sample number from {} to {} due to memory constraint'.format(TrainX.shape[0], m))
TrainX = TrainX[:m, :]
TrainY = TrainY[:m]
encodeX_train = encodingX(TrainX, modelpath)
del TrainX

print('Expanding encoded Training data into LRSC format')
encodeX_train_sc = preprocess_input_data(encodeX_train)
del encodeX_train
print('Fitting LRSC_encode model')
lrsc_encode = LogisticRegression(solver='newton-cg', max_iter=100)
lrsc_encode.fit(encodeX_train_sc, TrainY.ravel())
score_train_lrsc_encode = lrsc_encode.score(encodeX_train_sc, TrainY.ravel())
del encodeX_train_sc
print('LRSC_encode Model fit accuracy: {:.2f}'.format(score_train_lrsc_encode*100))

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

# print('raw LRSC Model accuracy: {:.2f}/{:.2f}/{:.2f}'.format(score_train_lrsc_raw*100,
#                                                              score_valid_lrsc_raw*100,
#                                                              score_test_lrsc_raw*100))

print('LRSC_encode Model accuracy: {:.2f}/{:.2f}/{:.2f}'.format(score_train_lrsc_encode*100,
                                                                score_valid_lrsc_encode*100,
                                                                score_test_lrsc_encode*100))
