import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class CostToGoNet(nn.Module):
    def __init__(self, input_size=48, hidden_size_1=1000, hidden_size_2=100, num_residual_blocks=2):
        super(CostToGoNet, self).__init__()
        self.initial_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU()
        )
        self.projection = nn.Linear(hidden_size_2, hidden_size_2)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size_2) for _ in range(num_residual_blocks)]
        )

        self.output_layer = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.projection(x)
        x = self.residual_blocks(x)
        return self.output_layer(x)
