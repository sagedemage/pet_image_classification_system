"""Structure of the Model"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class PetClassifier(nn.Module):
    """Define a Model for Pet Classification"""

    def __init__(self):
        super(PetClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.pool = nn.MaxPool2d(2, 8)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x =  torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.reshape(x, (4,))
        return x
