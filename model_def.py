import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 360 + 8
N_ACTIONS = 8

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)