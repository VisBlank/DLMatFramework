# References
# https://github.com/vinhkhuc/PyTorch-Mini-Tutorials
# https://arxiv.org/pdf/1604.07316.pdf

import torch.nn as nn

class CNNDriver(nn.Module):
    def __init__(self):
        super(CNNDriver, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv_1", nn.Conv2d(3, 24, kernel_size=5, padding=0, stride=2))
        self.layer1.add_module("relu_1", nn.ReLU())

        self.layer2 = nn.Sequential()
        self.layer2.add_module("conv_2", nn.Conv2d(24, 36, kernel_size=5, padding=0, stride=2))
        self.layer2.add_module("relu_2", nn.ReLU())

        self.layer3 = nn.Sequential()
        self.layer3.add_module("conv_3", nn.Conv2d(36, 48, kernel_size=5, padding=0, stride=2))
        self.layer3.add_module("relu_3", nn.ReLU())

        self.layer4 = nn.Sequential()
        self.layer4.add_module("conv_4", nn.Conv2d(48, 64, kernel_size=3, padding=0, stride=1))
        self.layer4.add_module("relu_4", nn.ReLU())

        self.layer5 = nn.Sequential()
        self.layer5.add_module("conv_5", nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1))
        self.layer5.add_module("relu_5", nn.ReLU())

        self.fc1 = nn.Sequential()
        self.fc1.add_module("fc1", nn.Linear(1152, 1164))
        self.fc1.add_module("relu_6", nn.ReLU())
        self.fc1.add_module("dropout_1", nn.Dropout(p=0.8))

        self.fc2 = nn.Sequential()
        self.fc2.add_module("fc2", nn.Linear(1164, 100))
        self.fc2.add_module("relu_7", nn.ReLU())
        self.fc2.add_module("dropout_2", nn.Dropout(p=0.8))

        self.fc3 = nn.Sequential()
        self.fc3.add_module("fc3", nn.Linear(100, 50))
        self.fc3.add_module("relu_8", nn.ReLU())
        self.fc3.add_module("dropout_3", nn.Dropout(p=0.8))

        self.fc4 = nn.Sequential()
        self.fc4.add_module("fc4", nn.Linear(50, 10))
        self.fc4.add_module("relu_9", nn.ReLU())
        self.fc4.add_module("dropout_4", nn.Dropout(p=0.8))

        self.fc_out = nn.Linear(10, 1)

    def forward(self, x):
        # The expected image size is 66x200
        layer1 = self.layer1.forward(x)
        layer2 = self.layer2.forward(layer1)
        layer3 = self.layer3.forward(layer2)
        layer4 = self.layer4.forward(layer3)
        layer5 = self.layer5.forward(layer4)

        # Reshape layer5 activation to a vector
        layer5_reshape = layer5.view(layer5.size(0), -1)

        fc1 = self.fc1.forward(layer5_reshape)
        fc2 = self.fc2.forward(fc1)
        fc3 = self.fc3.forward(fc2)
        fc4 = self.fc4.forward(fc3)

        fc_out = self.fc_out(fc4)
        return fc_out
