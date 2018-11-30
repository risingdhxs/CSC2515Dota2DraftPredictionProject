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

num_epochs = 50
batch_size = 128
learning_rate = 1e-3
model = autoencoder()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

print('Loading noleave_N dataset...')
dataset=np.load('../data/all/all_IO_noleave_N.npz')
TrainX=np.asarray(np.asmatrix(dataset['TrainX'])[0,0].astype(np.float32).todense())
ValidX=np.asarray(np.asmatrix(dataset['ValidX'])[0,0].astype(np.float32).todense())
TestX=np.asarray(np.asmatrix(dataset['TestX'])[0,0].astype(np.float32).todense())
print('Finished loading dataset')

print('Converting dataset matrices to torch tensors...')
n=TrainX.shape[1]
TrainXae=torch.from_numpy(np.vstack((TrainX[:,:int(n/2)],TrainX[:,int(n/2):])))
ValidXae=torch.from_numpy(np.vstack((ValidX[:,:int(n/2)],ValidX[:,int(n/2):])))
TestXae=torch.from_numpy(np.vstack((TestX[:,:int(n/2)],TestX[:,int(n/2):])))


dataloader=DataLoader(TrainXae, batch_size=batch_size, shuffle=True)

print('Training...')
print('Training Set size:',TrainXae.shape[0],' batchsize:',batch_size,' Chunks per epoch:', np.floor(TrainXae.shape[0]/batch_size))
for epoch in range(num_epochs):
    print('epoch:', epoch)
    i = 0
    start=time.time()
    for data in dataloader:
        i += 1
        if i % 1000 == 0:
            print(i, end=' ')
            sys.stdout.flush()

        encoding, recon = model(data)
        loss = criterion(recon, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    end=time.time()
#     print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data[0]))
    if epoch % 1 == 0:
        print('Finished epoch [{}/{}], training time {:.2f}s'.format(epoch + 1, num_epochs,end-start))
        encodingTrain, reconTrain = model(TrainXae)
        lossTrain = criterion(reconTrain, TrainXae)
        encodingValid, reconValid = model(ValidXae)
        lossValid = criterion(reconValid, ValidXae)
        encodingTest, reconTest = model(TestXae)
        lossTest = criterion(reconTest, TestXae)
        print('Training loss:{:.4f}, Validation loss:{:.4f}, Test loss:{:.4f}'.format(lossTrain.item(),lossValid.item(),lossTest.item()))

print('Finished training on all_IO_noleave_N dataset')
print('Saving Model...')
torch.save(model, 'AutoEncoder_100_50_10_50epoch.pt')
