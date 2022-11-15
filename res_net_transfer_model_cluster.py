import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from data import data
import augment_data

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Normalize(mean, std)
    ]),
}

# train_dataset, test_dataset = augment_data.create_split(test_transform = augment_data.Norm(), train_transform = augment_data.Norm())
# train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


train_dataset, test_dataset, train_dataloader, test_dataloader = data.load_datasets()
image_datasets = {
    'train' : train_dataset,
    'val'   : test_dataset
}
dataloaders = {
    'train' : train_dataloader,
    'val'   : test_dataloader
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

weights = data.sampler(image_datasets['train'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ['0', '1', '2', '3']



def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = data_transforms[phase](inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                # scheduler.step()
                pass

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


#### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.

# model = models.resnext50_32x4d(pretrained=True)

model = torch.load('resnextbig')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, num_epochs=40)
torch.save(model, '244sizeresnext2')


## This is if you want to freeze weights

# model_conv = torchvision.models.resnet152(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False

# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, 4)

# model_conv = model_conv.to(device)

# criterion = nn.CrossEntropyLoss()

# # Observe that only parameters of final layer are being optimized as
# # opposed to before.
# optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)

# # Decay LR by a factor of 0.1 every 7 epochs
# # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# model_conv = train_model(model_conv, criterion, optimizer_conv, num_epochs=25)

# torch.save(model_conv, './resnet_adam_frozen_152_25epochs.pth')

# testing 

# model = torch.load('./resnet_adam_frozen_152.pth')
# model.train()
# with torch.no_grad():
#     n_correct = 0
#     n_samples = dataset_sizes['val']
#     n_class_correct = [0 for i in range(4)]
#     n_class_samples = [0 for i in range(4)]
#     for images, labels in dataloaders['val']:
#         images = data_transforms['val'](images)
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs, 1)
#         # n_samples += labels.size(0)
#         n_correct += torch.sum(predicted == labels.data)
#         for i in range(64):
#             try:
#                 label = labels[i]
#                 pred = predicted[i]
#                 if (label == pred):
#                     n_class_correct[label] += 1
#                 n_class_samples[label] += 1
#             except:
#                 pass

#     acc = 100.0 * n_correct.double() / n_samples
#     print(f'Accuracy of the network: {acc} %')

#     for i in range(4):
#         acc = 100.0 * n_class_correct[i] / n_class_samples[i]
#         print(f'Accuracy of {classes[i]}: {acc} %')
#     print(n_class_samples)