import copy
import torch
import os
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('a', type=int) # fc id
parser.add_argument('p', type=str) # path
parser.add_argument('b', type=int) # batch size
parser.add_argument('d', type=int) # which data
args = parser.parse_args()
if args.d and args.d == 1:
    dataset=np.load('../../../../data/all/all_IO_noleave_N.npz')
else:
    dataset=np.load('../../../../data/all/all_IO_noleave_VH.npz')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = args.p

fc_id = args.a
if fc_id == 1:
    from networks import Net_1
    model = Net_1(226, 2).to(device)
elif fc_id == 2:
    from networks import Net_2
    model = Net_2(226, 2).to(device)
elif fc_id == 3:
    from networks import Net_3
    model = Net_3(226, 2).to(device)
elif fc_id == 4:
    from networks import Net_4
    model = Net_4(226, 2).to(device)
elif fc_id == 5:
    from networks import Net_5
    model = Net_5(226, 2).to(device)
elif fc_id == 6:
    from networks import Net_6
    model = Net_6(226, 2).to(device)
elif fc_id == 7:
    from networks import Net_7
    model = Net_7(226, 2).to(device)
elif fc_id == 8:
    from networks import Net_8
    model = Net_8(226, 2).to(device)
elif fc_id == 9:
    from networks import Net_9
    model = Net_9(226, 2).to(device)
elif fc_id == 10:
    from networks import Net_10
    model = Net_10(226, 2).to(device)
elif fc_id == 11:
    from networks import Net_11
    model = Net_11(226, 2).to(device)
elif fc_id == 12:
    from networks import Net_12
    model = Net_12(226, 2).to(device)

model.load_state_dict(torch.load(PATH))



loss = torch.nn.CrossEntropyLoss()






class LoadDotaDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, item):
        feature = self.x[item]
        label = self.y[item]
        return feature, label
    def __len__(self):
        return len(self.x)



testX=np.asmatrix(dataset['TestX'])[0,0].astype(np.float32).toarray()
testY=np.array(np.asmatrix(dataset['TestY'])[0,0].astype(np.int).toarray()).flatten()


testset = LoadDotaDataset(x=testX, y=testY)
test_loader = DataLoader(dataset=testset, batch_size=args.b, shuffle=False)



def testAc(model, x, y, device):
    x, y = x.to(device), y.to(device)
    fx = model(x)
    return computeAccuracy(fx, y)

def testLo(model, x, y, device):
    x, y = x.to(device), y.to(device)
    fx = model(x)
    output = loss(fx, y)
    l = output.item()
    return l


def computeAccuracy(fx, y):
    prediction = torch.max(F.softmax(fx), 1)[1]
    pred_y = prediction.data.cpu().numpy()
    target_y = y.data.cpu().numpy()
    accuracy = sum(pred_y == target_y)/len(pred_y)
    return accuracy

valid_acc, valid_loss, valid_batch = 0, 0, 0
with open(os.path.join('fc_'+str(fc_id)+'.batch.test.log'),'a') as f:
    for valid_data, valid_label in test_loader:
        valid_acc += testAc(model, valid_data, valid_label, device)
        valid_batch += 1
    valid_acc /= valid_batch
    valid_batch = 0
    for valid_data, valid_label in test_loader:
        valid_loss += testLo(model, valid_data, valid_label, device)
        valid_batch += 1
    valid_loss /= valid_batch

    f.write('accuracy:%f, loss:%f\n' % (valid_acc, valid_loss))
    f.flush()













