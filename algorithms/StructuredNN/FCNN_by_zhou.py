# -*- coding: utf-8 -*-
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import pickle

# -------------------------------------load the dataset---------------------------------
def load_data(name, number=None):
    """
    Dota2 normal level matches are stored in spare matrix.
    Retrieve the data with name key.
    'TrainX': training data inputs.
    'TrainY': training data targets.
    'TestX': test data inputs.
    'TestY': test data targets.
    'ValidX': validation data inputs.
    'ValidY': validation data targets.

    :param name: key of the data to retrieve.
    :return: np array
    """
    all_data = np.load('../../data/all/all_IO_noleave_N.npz')
    target_data = np.asmatrix(all_data[name])[0, 0]
    target_data = target_data.astype(np.float32)
    target_data = target_data[:number].toarray()

    return target_data


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

# -----------------------------------create NN and training------------------------------
# N is batch size; D_in is input dimension;
# H1 is hidden layer 1dimension;
# H2 is hidden layer 1dimension;
# H3 is hidden layer 1dimension;
# H4 is hidden layer 1dimension;
# D_out is output dimension.
N, D_in, H1, H2, H3, H4, D_out = None, 226, 50, 50, 100, 50, 1
learning_rate = 1e-4
batch_size = 5000
epochs_num = 500

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2),
    # torch.nn.ReLU(),
    # torch.nn.Linear(H2, H3),
    # torch.nn.ReLU(),
    # torch.nn.Linear(H3, H4),
    # torch.nn.ReLU(),
    torch.nn.Linear(H2, D_out),
    torch.nn.Sigmoid()
)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

model.cuda()
criterion.cuda()

train_x = load_data('TrainX', N)
train_y = load_data('TrainY', N)

valid_x = load_data('ValidX', N)
valid_y = load_data('ValidY', N)

test_x = load_data('TestX', N)
test_y = load_data('TestY', N)

trainset = LoadDotaDataset(x=train_x, y=train_y)
validset = LoadDotaDataset(x=valid_x, y=valid_y)
testset = LoadDotaDataset(x=test_x, y=test_y)

train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)


def check_validation_score(model):
    valid_loader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=False)
    result = np.zeros((0,1))

    num_of_batch = 0
    epoch_loss_sum = 0

    for valid_data, valid_label in valid_loader:
        valid_data = valid_data.cuda()
        valid_label = valid_label.cuda()
        out = model(valid_data)
        result = np.vstack((result, out.cpu().data.numpy()))
        loss = criterion(out, valid_label)
        epoch_loss_sum += loss.item()
        num_of_batch += 1

    epoch_avg_loss = epoch_loss_sum*1.0/num_of_batch

    # valid_score = accuracy_score(valid_y, result)
    print(result)
    result_binary = np.where(result > 0.5, 1, 0)

    print(np.hstack((result, result_binary, valid_y))[:10, :])
    valid_score = (result_binary == valid_y).sum()*1.0/valid_y.shape[0]

    return epoch_avg_loss, valid_score


def check_test_score(model):
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
    result = np.zeros((0, 1))

    num_of_batch = 0
    epoch_loss_sum = 0

    for test_data, test_label in test_loader:
        test_data = test_data.cuda()
        test_label = test_label.cuda()
        out = model(test_data)
        result = np.vstack((result, out.cpu().data.numpy()))
        loss = criterion(out, test_label)
        epoch_loss_sum += loss.item()
        num_of_batch += 1

    epoch_avg_loss = epoch_loss_sum * 1.0 / num_of_batch

    result_binary = np.where(result > 0.5, 1, 0)
    test_score = (result_binary == test_y).sum() * 1.0 / test_y.shape[0]

    return epoch_avg_loss, test_score

train_loss_list = []
valid_loss_dict = {}

for epoch in range(epochs_num):
    num_of_batch = 0
    epoch_loss_sum = 0
    for train_data, train_label in train_loader:
        train_data = train_data.cuda()
        train_label = train_label.cuda()

        out = model(train_data)

        loss = criterion(out, train_label)

        epoch_loss_sum += loss.item()
        num_of_batch += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_avg_loss = epoch_loss_sum*1.0/num_of_batch

    # if epoch % 10 == 1:
    # BUG, should choose best model to evaluate, this is fixed in structured NN
    valid_loss, valid_score = check_validation_score(model)
    valid_loss_dict[epoch] = [valid_loss, valid_score]
    train_loss_list.append(epoch_avg_loss)
    print('Epoch {} validation accuracy: {}, validation loss: {}, training loss: {}'.format(epoch, valid_score, valid_loss, epoch_avg_loss))
    # else:
    #     train_loss_list.append(epoch_avg_loss)
    #     print('loss: {}'.format(epoch_avg_loss))


test_loss, test_score = check_test_score(model)
print('Test accuracy: {}, test loss {}'.format(test_score, test_loss))

pickle.dump(model, open('NN_model.m', 'wb'))

final_result = {
    "train": train_loss_list,
    "valid": valid_loss_dict,
    "test": [test_loss, test_score]
}
pickle.dump(model, open('NN_model_result.m', 'wb'))