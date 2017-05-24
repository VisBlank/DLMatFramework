import torch.nn as nn
import torch.nn.functional as F


class CNNDriver(nn.Module):
    def __init__(self):
        super(CNNDriver, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=0, stride=2),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=5, padding=0, stride=2),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=5, padding=0, stride=2),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=0, stride=1),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.ReLU())

        self.fc1 = nn.Linear(1152, 1164)

        self.fc2 = nn.Linear(1164, 100)

        self.fc3 = nn.Linear(100, 50)

        self.fc4 = nn.Linear(50, 10)

        self.out = nn.Linear(10, 1)

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer2(layer2)
        layer4 = self.layer2(layer3)
        layer4_reshape = layer4.view(layer4.size(0), -1)

        fc1 = F.relu(self.fc1(layer4_reshape))
        fc1 = F.dropout(fc1, training=self.training)

        fc2 = F.relu(self.fc2(fc1))
        fc2 = F.dropout(fc2, training=self.training)

        fc3 = F.relu(self.fc3(fc2))
        fc3 = F.dropout(fc3, training=self.training)

        fc4 = F.relu(self.fc4(fc3))
        fc4 = F.dropout(fc4, training=self.training)

        out = self.out(fc4)
        return out
