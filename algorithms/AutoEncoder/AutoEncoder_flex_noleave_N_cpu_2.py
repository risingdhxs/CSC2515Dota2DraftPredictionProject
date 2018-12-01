import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from autoencoder_def_flex import autoencoder

num_epochs = 300
batch_size = 4096
learning_rate = 1e-3
n_print = 1
layer = [200, 100, 30, 5]
# model = autoencoder().cuda()
# criterion = nn.BCELoss().cuda()
model = autoencoder(layer)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

print('Loading noleave_N dataset...')
dataset = np.load('../../data/all/all_IO_noleave_N.npz')
TrainX = np.asarray(np.asmatrix(dataset['TrainX'])[0, 0].astype(np.float32).todense())
ValidX = np.asarray(np.asmatrix(dataset['ValidX'])[0, 0].astype(np.float32).todense())
print('Finished loading dataset')

print('Converting dataset matrices to torch tensors...')
n = TrainX.shape[1]
# TrainXae = torch.from_numpy(np.vstack((TrainX[:, :int(n / 2)], TrainX[:, int(n / 2):]))).cuda()
TrainXae = torch.from_numpy(np.vstack((TrainX[:, :int(n / 2)], TrainX[:, int(n / 2):])))
del TrainX
dataloader_train = DataLoader(TrainXae, batch_size=batch_size, shuffle=True)

# ValidXae = torch.from_numpy(np.vstack((ValidX[:, :int(n / 2)], ValidX[:, int(n / 2):]))).cuda()
ValidXae = torch.from_numpy(np.vstack((ValidX[:, :int(n / 2)], ValidX[:, int(n / 2):])))
del ValidX
dataloader_valid = DataLoader(ValidXae, batch_size=batch_size, shuffle=False)

if len(layer) == 3:
    print('Training AutoEncoder with layer {}-{}-{} with max epoch {}'.format(layer[0], layer[1], layer[2], num_epochs))
else:
    print('Training AutoEncoder with layer {}-{}-{}-{} with max epoch {}'.format(layer[0], layer[1], layer[2], layer[3], num_epochs))

print('Training Set size:', TrainXae.shape[0], ' batchsize:', batch_size, ' Chunks per epoch:',
      np.floor(TrainXae.shape[0] / batch_size))
loss_train = np.zeros((num_epochs, 1))
loss_valid = np.zeros((num_epochs, 1))

loss_opt = 1
epoch_opt = 0
model_state_dict_opt = model.state_dict()
optimizer_state_dict_opt = optimizer.state_dict()

for epoch in range(num_epochs):
    start = time.time()
    i = 0
    loss_sum = 0
    for data in dataloader_train:
        i += 1
        # if i % 500 == 0:
        #     print(i, end=' ')
        #     sys.stdout.flush()

        _, recon = model(data)
        loss = criterion(recon, data)
        loss_sum = loss_sum + loss.item()
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    # print(i)

    if epoch % n_print == 0:
        loss_train[epoch, 0] = loss_sum / i

        i = 0
        loss = 0
        for data in dataloader_valid:
            i += 1
            _, recon = model(data)
            loss = loss + criterion(recon, data).item()
        # loss_valid = np.vstack((loss_valid,loss/i))
        loss_valid[epoch, 0] = loss / i
        end = time.time()
        print(
            'Finished epoch [{}/{}], training time {:.2f}s/epoch. Training loss:{:.4f}, Validation loss:{:.4f}'.format(
                epoch + 1, num_epochs, end - start / n_print, loss_train[epoch, 0], loss_valid[epoch, 0]))
        if loss_valid[epoch, 0] < loss_opt:
            loss_opt = loss_valid[epoch,0]
            epoch_opt = epoch
            model_state_dict_opt = model.state_dict()
            optimizer_state_dict_opt = optimizer.state_dict()


TestX = np.asarray(np.asmatrix(dataset['TestX'])[0, 0].astype(np.float32).todense())
# TestXae = torch.from_numpy(np.vstack((TestX[:, :int(n / 2)], TestX[:, int(n / 2):]))).cuda()
TestXae = torch.from_numpy(np.vstack((TestX[:, :int(n / 2)], TestX[:, int(n / 2):])))
del TestX
dataloader_test = DataLoader(TestXae, batch_size=batch_size, shuffle=False)

i = 0
loss_test = 0
for data in dataloader_test:
    i += 1
    _, recon = model(data)
    loss_test = loss_test + criterion(recon, data).item()

loss_test = loss_test / i
filetitle='AutoEncoder_{}_{}_{}_all_IO_noleave_N.pt'.format(layer[0],layer[1],layer[2])
print('Finished Training {}-{}-{} AutoEncoder on all_IO_noleave_N dataset: optimal Validation loss:{:.4f} at epoch:{};'
      ' Test Set loss:{:.4f}'.format(layer[0],layer[1],layer[2],loss_opt, epoch_opt, loss_test))

torch.save({
    'model_state_dict': model_state_dict_opt,
    'optimizer_state_dict': optimizer_state_dict_opt,
    'epoch_max': num_epochs,
    'epoch_optimal': epoch_opt,
    'loss_optimal': loss_opt,
    'epoch_gap': n_print,
    'loss_train': loss_train,
    'loss_valid': loss_valid,
    'loss_test': loss_test,
}, filetitle)
print('Saved Model to',filetitle)