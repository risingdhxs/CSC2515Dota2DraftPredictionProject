import numpy as np
import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import sys

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(113, 100),nn.ReLU(True),
            nn.Linear(100, 50),nn.ReLU(True),
            nn.Linear(50, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 50),nn.ReLU(True),
            nn.Linear(50, 100),nn.ReLU(True),
            nn.Linear(100, 113),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


num_epochs = 100
batch_size = 512
learning_rate = 1e-3
n_print = 1
model = autoencoder().cuda()
criterion = nn.BCELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

print('Loading noleave_N dataset...')
dataset=np.load('../../data/all/all_IO_noleave_N.npz')
TrainX=np.asarray(np.asmatrix(dataset['TrainX'])[0,0].astype(np.float32).todense())
ValidX=np.asarray(np.asmatrix(dataset['ValidX'])[0,0].astype(np.float32).todense())
# TestX=np.asarray(np.asmatrix(dataset['TestX'])[0,0].astype(np.float32).todense())
print('Finished loading dataset')

print('Converting dataset matrices to torch tensors...')
n=TrainX.shape[1]
TrainXae = torch.from_numpy(np.vstack((TrainX[:,:int(n/2)],TrainX[:,int(n/2):]))).cuda()
del TrainX
dataloader_train=DataLoader(TrainXae, batch_size=batch_size, shuffle=True)

ValidXae = torch.from_numpy(np.vstack((ValidX[:,:int(n/2)],ValidX[:,int(n/2):]))).cuda()
del ValidX
dataloader_valid=DataLoader(ValidXae, batch_size=batch_size, shuffle=False)


print('Training...')
print('Training Set size:',TrainXae.shape[0],' batchsize:',batch_size,' Chunks per epoch:', np.floor(TrainXae.shape[0]/batch_size))
loss_train = np.zeros((num_epochs,1))
loss_valid = np.zeros((num_epochs,1))

for epoch in range(num_epochs):
    start=time.time()
    for data in dataloader_train:
        # i += 1
        # if i % 1000 == 0:
        #     print(i, end=' ')
        #     sys.stdout.flush()

        _, recon = model(data)
        loss = criterion(recon, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================

    if epoch % n_print == 0:
        i = 0
        loss = 0
        for data in dataloader_train:
            i += 1
            _, recon = model(data)
            loss = loss + criterion(recon, data).item()
        # loss_train = np.vstack((loss_train,loss/i))
        loss_train[epoch,0] = loss/i

        i = 0
        loss = 0
        for data in dataloader_valid:
            i += 1
            _, recon = model(data)
            loss = loss + criterion(recon, data).item()
        # loss_valid = np.vstack((loss_valid,loss/i))
        loss_valid[epoch,0] = loss/i
        end=time.time()
        print('Finished epoch [{}/{}], training time {:.2f}s/epoch. Training loss:{:.4f}, Validation loss:{:.4f}'.format(epoch + 1, num_epochs,end-start/n_print, loss_train[epoch,0],loss_valid[epoch,0]))

TestX=np.asarray(np.asmatrix(dataset['TestX'])[0,0].astype(np.float32).todense())
TestXae=torch.from_numpy(np.vstack((TestX[:,:int(n/2)],TestX[:,int(n/2):]))).cuda()
del TestX
dataloader_test=DataLoader(TestXae, batch_size=batch_size, shuffle=False)
i = 0
loss_test = 0
for data in dataloader_test:
    i += 1
    _, recon = model(data)
    loss_test = loss_test + criterion(recon, data).item()	

loss_test = loss/i
print('Finished Training, Test loss:{:.4f}'.format(loss_test))
	
print('Finished training on all_IO_noleave_N dataset')
print('Saving Model...')
torch.save(model, 'AutoEncoder_100_50_10_100epoch.pt')

import pickle
trainingdata = {'TrainingError': loss_train, 'ValidationError': loss_valid}
pickle.dump(trainingdata, open('AutoEncoder_traindata_100_50_10_100epoch.p','wb'))
