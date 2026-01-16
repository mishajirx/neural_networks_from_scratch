import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_sizes, dropout=0.0):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def model_01_linear(input_dim, num_classes):
    return MLP(input_dim, num_classes, hidden_sizes=[])


def model_02_small(input_dim, num_classes):
    return MLP(input_dim, num_classes, hidden_sizes=[32])


def model_03_medium(input_dim, num_classes):
    return MLP(input_dim, num_classes, hidden_sizes=[64, 32])


def model_04_large(input_dim, num_classes):
    return MLP(input_dim, num_classes, hidden_sizes=[128, 64, 32])


def model_05_dropout(input_dim, num_classes, dropout=0.3):
    return MLP(input_dim, num_classes, hidden_sizes=[128, 64], dropout=dropout)


MODEL_FACTORY = {
    "01_linear": model_01_linear,
    "02_small": model_02_small,
    "03_medium": model_03_medium,
    "04_large": model_04_large,
    "05_dropout": model_05_dropout,
}
