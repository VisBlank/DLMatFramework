"""
References:
    https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/5_convolutional_net.py
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
from drive_dataset import DriveData
from torch.utils.data import DataLoader
from model import CNNDriver

# Hyper Parameters
num_epochs = 10
batch_size = 4000
learning_rate = 0.01
L2NormConst = 0.001

cnn = CNNDriver()
# Put model on GPU
cnn = cnn.cuda()

transformations = transforms.Compose([
    transforms.ToTensor()])

# Instantiate a dataset
dset_train = DriveData('./Track1_Wheel/', transformations)
dset_train.addFolder('./Track2_Wheel/')
dset_train.addFolder('./Track3_Wheel/')
dset_train.addFolder('./Track4_Wheel/')
dset_train.addFolder('./Track5_Wheel/')
dset_train.addFolder('./Track6_Wheel/')
dset_train.addFolder('./Track7_Wheel/')

train_loader = DataLoader(dset_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=3, # 1 for CUDA
                          pin_memory=True # CUDA only
                         )


# Loss and Optimizer
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=L2NormConst)

print('Train size:',len(dset_train), 'Batch size:', batch_size)
print('Batches per epoch:',len(dset_train) // batch_size)
# Train the Model
for epoch in range(num_epochs):
    iteration_count = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = Variable(images, requires_grad=False)
        labels = Variable(labels, requires_grad=False)

        # Send images and labels to GPU
        images, labels = images.cuda(), labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        #loss = loss_func.forward(outputs, labels)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # Display on each epoch
        if batch_idx == 0:
            print('Epoch [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, loss.data[0]))
            # Save the Trained Model parameters
            torch.save(cnn.state_dict(), 'cnn_' + str(epoch) + '.pkl')

        iteration_count += 1


