from torch import nn

class autoencoder(nn.Module):
    def __init__(self, nodenum):
        super(autoencoder, self).__init__()
        if len(nodenum) == 3:
            self.encoder = nn.Sequential(
                nn.Linear(113, nodenum[0]), nn.ReLU(True),
                nn.Linear(nodenum[0], nodenum[1]), nn.ReLU(True),
                nn.Linear(nodenum[1], nodenum[2])
            )
            self.decoder = nn.Sequential(
                nn.Linear(nodenum[2], nodenum[1]), nn.ReLU(True),
                nn.Linear(nodenum[1], nodenum[0]), nn.ReLU(True),
                nn.Linear(nodenum[0], 113), nn.Sigmoid()
            )
        elif len(nodenum) == 4:
            self.encoder = nn.Sequential(
                nn.Linear(113, nodenum[0]), nn.ReLU(True),
                nn.Linear(nodenum[0], nodenum[1]), nn.ReLU(True),
                nn.Linear(nodenum[1], nodenum[2]), nn.ReLU(True),
                nn.Linear(nodenum[2], nodenum[3])
            )
            self.decoder = nn.Sequential(
                nn.Linear(nodenum[3], nodenum[2]), nn.ReLU(True),
                nn.Linear(nodenum[2], nodenum[1]), nn.ReLU(True),
                nn.Linear(nodenum[1], nodenum[0]), nn.ReLU(True),
                nn.Linear(nodenum[0], 113), nn.Sigmoid()
            )
        elif len(nodenum) == 6:
            self.encoder = nn.Sequential(
                nn.Linear(113, nodenum[0]), nn.ReLU(True),
                nn.Linear(nodenum[0], nodenum[1]), nn.ReLU(True),
                nn.Linear(nodenum[1], nodenum[2]), nn.ReLU(True),
                nn.Linear(nodenum[2], nodenum[3]), nn.ReLU(True),
                nn.Linear(nodenum[3], nodenum[4]), nn.ReLU(True),
                nn.Linear(nodenum[4], nodenum[5])
            )
            self.decoder = nn.Sequential(
                nn.Linear(nodenum[5], nodenum[4]), nn.ReLU(True),
                nn.Linear(nodenum[4], nodenum[3]), nn.ReLU(True),
                nn.Linear(nodenum[3], nodenum[2]), nn.ReLU(True),
                nn.Linear(nodenum[2], nodenum[1]), nn.ReLU(True),
                nn.Linear(nodenum[1], nodenum[0]), nn.ReLU(True),
                nn.Linear(nodenum[0], 113), nn.Sigmoid()
            )
        elif len(nodenum) == 10:
            self.encoder = nn.Sequential(
                nn.Linear(113, nodenum[0]), nn.ReLU(True),
                nn.Linear(nodenum[0], nodenum[1]), nn.ReLU(True),
                nn.Linear(nodenum[1], nodenum[2]), nn.ReLU(True),
                nn.Linear(nodenum[2], nodenum[3]), nn.ReLU(True),
                nn.Linear(nodenum[3], nodenum[4]), nn.ReLU(True),
                nn.Linear(nodenum[4], nodenum[5]), nn.ReLU(True),
                nn.Linear(nodenum[5], nodenum[6]), nn.ReLU(True),
                nn.Linear(nodenum[6], nodenum[7]), nn.ReLU(True),
                nn.Linear(nodenum[7], nodenum[8]), nn.ReLU(True),
                nn.Linear(nodenum[8], nodenum[9])
            )
            self.decoder = nn.Sequential(
                nn.Linear(nodenum[9], nodenum[8]), nn.ReLU(True),
                nn.Linear(nodenum[8], nodenum[7]), nn.ReLU(True),
                nn.Linear(nodenum[7], nodenum[6]), nn.ReLU(True),
                nn.Linear(nodenum[6], nodenum[5]), nn.ReLU(True),
                nn.Linear(nodenum[5], nodenum[4]), nn.ReLU(True),
                nn.Linear(nodenum[4], nodenum[3]), nn.ReLU(True),
                nn.Linear(nodenum[3], nodenum[2]), nn.ReLU(True),
                nn.Linear(nodenum[2], nodenum[1]), nn.ReLU(True),
                nn.Linear(nodenum[1], nodenum[0]), nn.ReLU(True),
                nn.Linear(nodenum[0], 113), nn.Sigmoid()
            )
        else:
            print('Currently not supporting encoding layer other than 3/4/6/10')

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded