import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.relu(out + identity)

class DeepCubeA(nn.Module):
    def __init__(self, input_size=48, hidden1=2000, hidden2=400, residual_blocks=2): #5000 1000
        super(DeepCubeA, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden2) for _ in range(residual_blocks)])
        self.output = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        for block in self.residual_blocks:
            x = block(x)
        return self.output(x)

# Funzione di utilità per inizializzare il modello
def initialize_model():
    model = DeepCubeA()
    return model
