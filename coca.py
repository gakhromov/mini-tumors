import torchvision
import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

import sys
from sys import path
import sys

import numpy as np

from data.data import load_datasets, Data
from augment_data import AugmentedData, create_split
import matplotlib.pyplot as plt


# device = torch.device("mps")
# Model
model = models.inception_v3(pretrained=True)

model.aux_logits = False

for parameter in model.parameters():
    parameter.requires_grad = False


# Warning: Inception v3 needs images of size  [batch_size, 3, 299, 299]

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

batch_size = 64

train_data, test_data = create_split(test_transform=test_transform, train_transform=train_transform, test_percentage=0.2)

#AugmentedData(train_transform), AugmentedData(test_transform)




test_loader = torch.utils.data.DataLoader(test_data, drop_last=True, batch_size=batch_size)
train_loader = torch.utils.data.DataLoader(train_data, drop_last=True, batch_size=batch_size) #drop_last=True to drop last wierdly sized batch

# Hmmmmm ...
model.aux_logits = False

# Data input in batch N, channels C, depth D, height H, width W
# expected input[1, 64, 3, 128]
#          Want     Depth   |  Batch Size  | Channels | Height |
#          Get        smt    |   smt       | Batch Size | Height| 

# OpenCV will spit things out as  Heigh Width Channels
# Whereas pytorch wants it as channels height width

# Freeze parameters
for parameter in model.parameters():
    parameter.requires_grad = False

# Reinitialize final fully connected layer for our 4 classes
# Original architecutre ends with
# (fc): Linear(in_features=2048, out_features=1000, bias=True)

# Add two layers so we can approximate an arbitrary function
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 10),
    nn.Linear(10, 4)
)

# Run on M1 Chip GPU
# model.to(device)

# TODO: Pick a less whack learning rate

# May be smart to assign weigths
# Distribution of classes in dataset is: [0.29561881, 0.49897386, 0.13152494, 0.07388239]
# weight_for_class_a = (1 / samples_for_class_a) * total_number_of_samples/number_of_classes
loss = nn.CrossEntropyLoss(weight=torch.tensor([0.8456836694525629, 0.5010282502574384, 1.9007801866322842, 3.38375626451716]))

optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)

num_epochs = 2

for epoch in range(num_epochs):
    print("Training Epoch no" + str(epoch))

    total_batch = train_loader.__len__()//batch_size

    # Buckets to keep track of how predicted labels are distributed
    freq_counts = np.array([0,0,0,0])

    for i, (batch_images, batch_labels) in enumerate(train_loader):
        
        #X = batch_images.cuda()
        #Y = batch_labels.cuda()

        X = batch_images
        Y = batch_labels

        # M1 being a bit difficult. Current build wont accept DoublePrecision tensors

        #X = X.to(device=device, dtype=torch.float32)
        #Y = Y.to(device=device, dtype=torch.float32)
        X = X.to(dtype=torch.float32)
        Y = Y.to(dtype=torch.int64)

        pre = model(X)
        cost = loss(pre, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        #freq_counts = freq_counts + np.unique(np.array(torch.max(pre, 1)[1]), return_counts=True)[1]

        if (i+1) % 5 == 0:
            print('Epoch [%d/%d], lter [%d/%d] Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, total_batch, cost.item()))
            #print('Class freuencies: ' + str( freq_counts / sum(freq_counts) ))
            print('Class freuencies: ' + labels)
            #freq_counts = np.array([0,0,0,0])
            pass


model.eval()

correct = 0
total = 0

for images, labels in test_loader:
    
    #images = images.cuda()
    #images.to(device)

    outputs = model(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    #correct += (predicted == labels.cuda()).sum()
    correct += (predicted == labels).sum()
    
print('Accuracy of test images: %f %%' % (100 * float(correct) / total))


classes = ["Squirrel", "Chipmunk"]


images, labels = iter(test_loader).next()

#outputs = model(images.cuda())
outputs = model(images)

_, predicted = torch.max(outputs.data, 1)
    
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(5)))

title = (' '.join('%5s' % classes[labels[j]] for j in range(5)))
plt.imshow(torchvision.utils.make_grid(images, normalize=True), title)


'''
    mps_device = torch.device("mps")

    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happens on the GPU
    y = x * 2

    # Move your model to mps just like any other device
    model = YourFavoriteNet()
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(x)

'''