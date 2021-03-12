# RanDair Porter and Doron Barasch

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.fc1 = nn.Linear(4096, 28)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

