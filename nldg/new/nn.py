import torch.nn as nn


class NN(nn.Module):
    def __init__(self, input_dim):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


class NN_GDRO(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64]):
        """
        Neural network with configurable hidden layers.

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list of int): Sizes of hidden layers.
        """
        super(NN_GDRO, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
