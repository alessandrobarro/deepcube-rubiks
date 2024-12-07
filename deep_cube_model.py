import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block with two hidden layers as described in the DeepCubeA paper.
    """
    def __init__(self, in_channels, hidden_channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.batch_norm2(self.fc2(x))
        return F.relu(x + residual)

class DeepCubeAModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_residual_blocks=4):
        super(DeepCubeAModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, hidden_size) for _ in range(num_residual_blocks)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)  # Single output for cost-to-go

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)
