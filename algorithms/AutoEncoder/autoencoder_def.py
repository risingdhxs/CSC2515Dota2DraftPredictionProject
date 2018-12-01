import torch
from torch import nn


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(113, 100), nn.ReLU(True),
            nn.Linear(100, 50), nn.ReLU(True),
            nn.Linear(50, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 50), nn.ReLU(True),
            nn.Linear(50, 100), nn.ReLU(True),
            nn.Linear(100, 113),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded