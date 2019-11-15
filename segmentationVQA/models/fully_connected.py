import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class fully_connected_network(nn.Module):

    def __init__(self, dims):
        super(fully_connected_network, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(weight_norm(nn.Linear(dims[i], dims[i+1]), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
