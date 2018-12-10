import torch
import numpy as np

from torch.autograd import Variable

import torch.nn.functional as F

class Net_1(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_1, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 100),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x

class Net_2(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_2, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 100),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x


class Net_3(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_3, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 160),
            torch.nn.ReLU(True),
            torch.nn.Linear(160, 100),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x

class Net_4(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_4, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 160),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(160, 100),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x

class Net_5(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_5, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 160),
            torch.nn.ReLU(True),
            torch.nn.Linear(160, 100),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, 20),
            torch.nn.ReLU(True),
            torch.nn.Linear(20, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x

class Net_6(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_6, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 160),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(160, 100),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, 20),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(20, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x

class Net_7(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_7, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 10),
            torch.nn.ReLU(True),
            torch.nn.Linear(10, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x

class Net_8(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_8, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 10),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(10, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x


class Net_9(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_9, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 160),
            torch.nn.ReLU(True),
            torch.nn.Linear(160, 10),
            torch.nn.ReLU(True),
            torch.nn.Linear(10, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x

class Net_10(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_10, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 160),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(160, 10),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(10, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x

class Net_11(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_11, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 160),
            torch.nn.ReLU(True),
            torch.nn.Linear(160, 100),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x

class Net_12(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net_12, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feature, 160),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(160, 100),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, 100),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, n_output)
            )
    def forward(self, x):
        x = self.fc.forward(x)
        return x

