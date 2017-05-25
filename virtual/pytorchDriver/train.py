import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
from drive_dataset import DriveData
from torch.utils.data import DataLoader
from model import CNNDriver

# Hyper Parameters
num_epochs = 5
batch_size = 50
learning_rate = 0.001

cnn = CNNDriver()
# Put model on GPU
cnn = cnn.cuda()

transformations = transforms.Compose([
    transforms.RandomCrop((66, 200)),transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

# Instantiate a dataset
dset_train = DriveData('./Track1_Wheel/', transformations)

train_loader = DataLoader(dset_train,
                          batch_size=400,
                          shuffle=True,
                          num_workers=1, # 1 for CUDA
                          pin_memory=True # CUDA only
                         )


# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

print('Train size:',len(dset_train), 'Batch size:', 400)
print('Batches per epoch:',len(dset_train) // batch_size)
# Train the Model
for epoch in range(num_epochs):
    iteration_count = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)

        # Send images and labels to GPU
        images, labels = images.cuda(), labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print('Epoch [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, loss.data[0]))

        iteration_count += 1