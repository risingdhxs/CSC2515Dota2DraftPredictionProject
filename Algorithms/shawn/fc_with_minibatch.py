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
parser.add_argument('e', type=int) # epochs
parser.add_argument('p', type=str) # path
parser.add_argument('b', type=int) # batch size
parser.add_argument('d', type=int) # which data



args = parser.parse_args()

fc_id = args.a
epochs = args.e

if args.d and args.d == 1:
    dataset=np.load('../../data/all/all_IO_noleave_N.npz')
else:
    dataset=np.load('../../data/all/all_IO_noleave_VH.npz')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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



loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

the_epoch, pre_acc, best_model = 0, 0, None # keep tracking the one giving best validation accuacy

path = os.path.abspath(args.p)


################################### for batching


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


############################################
#
# Observation: 23, 107, 136, 220 are heroes that are never used
#
############################################



trainX=np.asmatrix(dataset['TrainX'])[0,0].astype(np.float32).toarray()
trainY=np.array(np.asmatrix(dataset['TrainY'])[0,0].astype(np.int).toarray()).flatten()
validX=np.asmatrix(dataset['ValidX'])[0,0].astype(np.float32).toarray()
validY=np.array(np.asmatrix(dataset['ValidY'])[0,0].astype(np.int).toarray()).flatten()
testX=np.asmatrix(dataset['TestX'])[0,0].astype(np.float32).toarray()
testY=np.array(np.asmatrix(dataset['TestY'])[0,0].astype(np.int).toarray()).flatten()



trainset = LoadDotaDataset(x=trainX, y=trainY)
validset = LoadDotaDataset(x=validX, y=validY)
# testset = LoadDotaDataset(x=testX, y=testY)

train_loader = DataLoader(dataset=trainset, batch_size=args.b, shuffle=True)
valid_loader = DataLoader(dataset=validset, batch_size=args.b, shuffle=False)
# test_loader = DataLoader(dataset=testset, batch_size=args.b, shuffle=False)



# for train_data, train_label in train_loader:
#     print(train_label)


###################################



def train(model, loss, optimizer, x, y, device):
    x, y = x.to(device), y.to(device)
    fx = model(x)
    output = loss(fx, y)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
    # return computeAccuracy(fx, y)
    return computeLoss(fx, y)


def test(model, x, y, device):
    x, y = x.to(device), y.to(device)
    fx = model(x)
    output = loss(fx, y)
    l = output.item()
    # return computeAccuracy(fx, y)
    return computeLoss(fx, y)

 
def computeAccuracy(fx, y):
    prediction = torch.max(F.softmax(fx), 1)[1]
    pred_y = prediction.data.cpu().numpy()
    target_y = y.data.cpu().numpy()
    accuracy = sum(pred_y == target_y)/len(pred_y)
    return accuracy

def computeLoss(fx, y):
    _loss = torch.nn.CrossEntropyLoss()
    l = _loss(fx, y).item()
    return l



with open(os.path.join(path,'fc_'+str(fc_id)+'.batch.log'),'a') as f:
    for i in range(epochs):
        train_batch, train_acc, valid_batch, valid_acc = 0, 0, 0, 0
        for train_data, train_label in train_loader:
            train_acc += train(model, loss, optimizer, train_data, train_label, device)
            train_batch += 1
        train_acc /= train_batch
        for valid_data, valid_label in valid_loader:
            valid_acc += test(model, valid_data, valid_label, device)
            valid_batch += 1
        valid_acc /= valid_batch
        if valid_acc > pre_acc:
            best_model = copy.deepcopy(model)
            pre_acc = valid_acc
            the_epoch = i
        f.write('epoch %d/%d,training loss:%f,validation loss:%f\n' % (i, epochs, train_acc, valid_acc))
        f.flush()
        
    model_path = os.path.join(path,'fc_'+str(fc_id)+'.batch.pth')
    torch.save(best_model.state_dict(), model_path)        
    f.write('Finished. Saving the model to %s\n' % model_path)
    f.write('The best validation accuracy occurs at %dth epoch' % (the_epoch))











