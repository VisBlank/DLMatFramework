"""
References:
    https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/5_convolutional_net.py
"""
from visdom import Visdom
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
from drive_dataset import DriveData
from torch.utils.data import DataLoader
from model import CNNDriver

# Initialize Visdom
viz = Visdom()
hist_labels = None

# Hyper Parameters
num_epochs = 1000
batch_size = 400
learning_rate = 0.01
L2NormConst = 0.001

cnn = CNNDriver()
print(cnn)
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

        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = cnn(images)
        #loss = loss_func(outputs, labels)
        loss = (outputs - labels).pow(2).sum()

        loss.backward()
        optimizer.step()

        # Display on each epoch
        if batch_idx == 0:
            print('Epoch [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, loss.data[0]))

            # Close previous window
            if hist_labels != None:
                viz.close(win=hist_labels)
                viz.close(win=hist_output)
                viz.close(win=grid_batch)

            # Vizualize some stuff
            hist_labels = viz.histogram(labels.data.cpu().numpy(), opts=dict(title='Labels histogram epoch='+str(epoch)))
            hist_output = viz.histogram(outputs.data.cpu().numpy(), opts=dict(title='Output histogram epoch='+str(epoch)))
            video_visdom = images.data.cpu().numpy() * 255.0
            grid_batch = viz.images(video_visdom[:16], opts=dict(title='Batch images sample'))
            #viz.video(video_visdom)

            # Save the Trained Model parameters
            torch.save(cnn.state_dict(), 'cnn_' + str(epoch) + '.pkl')

        iteration_count += 1


