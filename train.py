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
import argparse
import tqdm as tqdm

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-ne", "--n-epochs", type=int, default=15,
                    help="number of epochs to train the model for")
parser.add_argument("-mf", "--model-filename", type=str, default="saved_models/best_model.pt",
                    help="filename under which to save the best model")
args = parser.parse_args()

# Load data
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

# Train model
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model = copy.deepcopy(model)
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
            for inputs, labels in enumerate(tqdm(dataloaders[phase])):
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
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model, best_model

model = models.resnext101_32x8d(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model, best_model = train_model(model, criterion, optimizer, num_epochs=args.n_epochs)

# Save best model
torch.save(best_model, args.model_filename)