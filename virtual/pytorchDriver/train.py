import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
from model import CNNDriver

# Hyper Parameters
num_epochs = 5
batch_size = 50
learning_rate = 0.001

cnn = CNNDriver()
print(cnn)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)