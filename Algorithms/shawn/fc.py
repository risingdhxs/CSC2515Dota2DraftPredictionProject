import copy
import torch
import os
import numpy as np

from torch.autograd import Variable

import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('a', type=int) # fc id
parser.add_argument('e', type=int) # epochs
parser.add_argument('p', type=str) # path


args = parser.parse_args()

fc_id = args.a
epochs = args.e


dataset=np.load('data/all/all_IO_noleave_N.npz')

trainX=np.asmatrix(dataset['TrainX'])[0,0].todense().astype(np.float32)

validX=np.asmatrix(dataset['ValidX'])[0,0].todense().astype(np.float32)

testX=np.asmatrix(dataset['TestX'])[0,0].todense().astype(np.float32)

trainY=np.array(np.asmatrix(dataset['TrainY'])[0,0].todense().astype(np.float32)).flatten()

validY=np.array(np.asmatrix(dataset['ValidY'])[0,0].todense().astype(np.float32)).flatten()

testY=np.array(np.asmatrix(dataset['TestY'])[0,0].todense().astype(np.float32)).flatten()





def train(model, loss, optimizer, x, y, device):
    x, y = Variable(torch.from_numpy(x)).to(device), Variable(torch.from_numpy(y)).to(device)
    fx = model(x)
    output = loss(fx, y)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
    return computeAccuracy(fx, y)


def test(model, x, y, device):
    x, y = Variable(torch.from_numpy(x)).to(device), Variable(torch.from_numpy(y)).to(device)
    fx = model(x)
    return computeAccuracy(fx, y)

 
def computeAccuracy(fx, y):
    prediction = torch.max(F.softmax(fx), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    target_y = y.data.numpy()
    accuracy = sum(pred_y == target_y)/len(pred_y)
    return accuracy


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
    from networks import Net_4 as Net
    model = Net_4(226, 2).to(device)
elif fc_id == 5:
    from networks import Net_5
    model = Net_5(226, 2).to(device)
elif fc_id == 6:
    from networks import Net_6
    model = Net_6(226, 2).to(device)


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

the_epoch, pre_acc, best_model = 0, 0, None # keep tracking the one giving best validation accuacy

path = os.path.abspath(args.p)

with open(os.path.join(path,'fc_'+str(fc_id)+'.log'),'a') as f:
    for i in range(epochs):
        train_acc = train(model, loss, optimizer, trainX, trainY.astype(int), device)
        valid_acc = test(model, validX, validY.astype(int), device)
        if valid_acc > pre_acc:
            best_model = copy.deepcopy(model)
            the_epoch = i
        f.write('epoch %d/%d,training accuracy:%f,validation accuracy:%f\n' % (i, epochs, train_acc, valid_acc))
        f.flush()
    model_path = os.path.join(path,'fc_'+str(fc_id)+'.pth')
    torch.save(best_model.state_dict(), model_path)        
    f.write('Finished. Saving the model to %s\n' % model_path)
    f.write('The best validation accuracy occurs at %dth epoch' % (the_epoch))

# from netwoks import Net_1 as Net
# the_model = Net_1(226,2)
# the_model.load_state_dict(torch.load(PATH))

# for name, param in the_model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)









