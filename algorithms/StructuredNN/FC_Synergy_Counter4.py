# -*- coding: utf-8 -*-
import datetime
import time
import copy
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
import pickle

OUTPUT_EXT='_lr0.001_113_200_200_400_400_100_1_noleave_N'

# -------------------------------------load the dataset---------------------------------
def load_data(name, n=None):
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
    :param n: number of data to load, load all if n is None
    :return: np array
    """
    all_data = np.load('../../data/all/all_IO_noleave_N.npz')
    target_data = np.asmatrix(all_data[name])[0, 0]
    target_data = target_data.astype(np.float32)
    target_data = target_data[:n].toarray()

    return target_data


class LoadDotaDataset(Dataset):
    """
    Dota dataset extends from torch Dataset.
    """
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
class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    D_in = 113  # number of features per team, 113

    # Synergy layer sizes
    S1 = 200
    S2 = 200

    # Counter layer sizes
    C1 = 400
    C2 = 400
    C3 = 100

    D_out = 1  # output dimension

    learning_rate = 1e-3
    batch_size = 10000
    epochs_size = 500

    def __str__(cls):
        return "Neural Network with six layers: {}, {}, {}, {}, {}, {}. Learning rate: {}, epochs: {}".format(
            cls.S1, cls.S2, cls.C1, cls.C2, cls.C3, cls.D_out, cls.learning_rate, cls.epochs_size
        )


class DotaNet(torch.nn.Module):
    def __init__(self, config):
        """
        In the constructor we construct nn.Linear instances that we will use in the forward pass.
        :param config: instance of class Config.
        """
        super(DotaNet, self).__init__()

        # Two layers synergy network on One Team
        self.synergy_net = torch.nn.Sequential(
            torch.nn.Linear(config.D_in, config.S1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.S1, config.S2),
            torch.nn.ReLU()
        )

        # layers of counter network
        self.counter_net = torch.nn.Sequential(
            torch.nn.Linear(config.S2*2, config.C1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(config.C1, config.C2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(config.C2, config.C3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(config.C3, config.D_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x_red, x_blue):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        :param x_red: red team inputs
        :param x_blue: blue team inputs
        """
        red_synergy_out = self.synergy_net(x_red)
        blue_synergy_out = self.synergy_net(x_blue)

        # cat two teams together
        cat_synergy = torch.cat((red_synergy_out, blue_synergy_out), 1)
        out = self.counter_net(cat_synergy)
        return out


def calculate_accuracy(pred, target):
    """
    Calculate accuracy of prediction based on target
    :param pred: np array
    :param target: np array
    :return: accuracy (float)
    """
    pred_binary = np.where(pred>0.5, 1, 0)
    score = (pred_binary == target).sum() / target.shape[0]
    return score


def evaluate_performance(model, criterion, dataset, target, config):
    """
    Evaluate model performance on the given dataset.
    :param model:
    :param criterion:
    :param dataset:
    :param config:
    :return: accuracy, cross-entropy loss
    """
    data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)
    all_y_pred = np.zeros((0, 1))

    batch_count = 0
    epoch_loss_sum = 0

    for data_x, data_y in data_loader:
        x_red, x_blue = torch.split(data_x, config.D_in, dim=1)

        # convert to GPU tensor
        x_red = x_red.cuda()
        x_blue = x_blue.cuda()
        data_y = data_y.cuda()

        y_pred = model(x_red, x_blue)  # predict y for training data
        all_y_pred = np.vstack((all_y_pred, y_pred.cpu().data.numpy()))
        loss = criterion(y_pred, data_y)  # loss for one batch

        epoch_loss_sum += loss.item()
        batch_count += 1

    epoch_avg_loss = epoch_loss_sum/batch_count

    score = calculate_accuracy(all_y_pred, target)

    return score, epoch_avg_loss


def save_model(model):
    print("Saving the model to NN_model.m ...")
    torch.save(model, 'NN_model{}.m'.format(OUTPUT_EXT))

    pickle.dump(model, open('NN_model{}.pickle'.format(OUTPUT_EXT), 'wb'))
    print("Finished saving...")


def startNN():
    # Load data and build Dataset
    train_x = load_data('TrainX')
    train_y = load_data('TrainY')

    valid_x = load_data('ValidX')
    valid_y = load_data('ValidY')

    test_x = load_data('TestX')
    test_y = load_data('TestY')

    trainset = LoadDotaDataset(x=train_x, y=train_y)
    validset = LoadDotaDataset(x=valid_x, y=valid_y)
    testset = LoadDotaDataset(x=test_x, y=test_y)

    # Construct our model by instantiating the class defined above
    config = Config()
    print(config)
    model = DotaNet(config)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    # Use GPU version
    model.cuda()
    criterion.cuda()

    train_loss_list = []
    train_accuracy_list = []
    valid_loss_list = []
    valid_accuracy_list = []

    best_valid_loss = None
    best_model = None

    # start training
    print("Start training...")
    train_loader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs_size):
        batch_count = 0  # count how many batches in this epoch
        epoch_loss_sum = 0

        all_y_pred = np.zeros((0, 1))

        for data_x, data_y in train_loader:
            x_red, x_blue = torch.split(data_x, config.D_in, dim=1)

            # convert to GPU tensor
            x_red = x_red.cuda()
            x_blue = x_blue.cuda()
            data_y = data_y.cuda()

            y_pred = model(x_red, x_blue)  # predict y for batch training data
            all_y_pred = np.vstack((all_y_pred, y_pred.cpu().data.numpy()))
            loss = criterion(y_pred, data_y)  # loss for one batch

            epoch_loss_sum += loss.item()
            batch_count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_avg_loss = epoch_loss_sum/batch_count
        train_score = calculate_accuracy(all_y_pred, train_y)

        train_loss_list.append(epoch_avg_loss)
        train_accuracy_list.append(train_score)

        valid_score, valid_loss = evaluate_performance(model, criterion, validset, valid_y, config)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_score)

        if best_valid_loss is None or valid_loss < best_valid_loss:
            best_model = copy.deepcopy(model)
            best_valid_loss = valid_loss

            # save this model to file
            save_model(best_model)

        print('Epoch {}, training accuracy: {:10.4f}, training loss: {:10.4f}, validation accuracy: {:10.4f}, '
              'validation loss: {:10.4f}'.format(epoch, train_score, epoch_avg_loss, valid_score, valid_loss))

        # save the performance to file periodically
        performances = {
            'train_loss': train_loss_list,
            'train_accuracy': train_accuracy_list,
            'valid_loss': valid_loss_list,
            'valid_accuracy': valid_accuracy_list,
        }
        pickle.dump(performances, open('NN_train_valid_performance{}.pickle'.format(OUTPUT_EXT), 'wb'))

    print("Evaluate the best model on test set...")
    test_score, test_loss = evaluate_performance(best_model, criterion, testset, test_y, config)
    print('Test accuracy: {:10.4f}, test loss {:10.4f}'.format(test_score, test_loss))

    pickle.dump({
        'test_loss': test_loss,
        'test_accuracy': test_score
    }, open('NN_test_performance{}.pickle'.format(OUTPUT_EXT), 'wb'))


startNN()
