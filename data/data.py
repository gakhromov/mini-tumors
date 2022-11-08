import os

import numpy as np
import pandas as pd
import torch
import nd2
import json
from skimage.transform import resize

import matplotlib.pyplot as plt

from data import config


class Data(torch.utils.data.Dataset):
    def __init__(self):
        self.labels = np.load(f'{config.ROOT_PATH}/data/clean/labels.npy')
        with open(f'{config.ROOT_PATH}/data/clean/samples.json', 'r') as f:
            self.sample_list = json.load(f)['samples']
        with open(f'{config.ROOT_PATH}/data/clean/droplets.json', 'r') as f:
            self.droplet_list = json.load(f)['droplets']
	
    def show_sample(self, sample_idx: int, channel: int = -1):
        if sample_idx >= len(self.sample_list):
            print(f'Error: index too large (there are {len(self.sample_list)} samples)')
            return

        img = nd2.imread(self.sample_list[sample_idx]['img_path'])

        if not ((0 <= channel <= img.shape[0]-1) or (channel == -1)):
            print(f'Error: there are only {img.shape[0]} channels to visualize. channel should be a value between 0 and {img.shape[0]-1}.')
            return

        if channel == -1:
            channel = img.shape[0]-1

        plt.title('Sample = ' + self.sample_list[sample_idx]['name'])
        plt.imshow(img[channel,:,:])
        plt.show()

    def show_droplet(self, idx: int, channel: int = -1):
        if idx >= len(self.labels):
            print(f'Error: index too large (there are {len(self.labels)} labeled droplets)')
            return

        label = self.labels[idx, 1]
        img = np.load(f'{config.ROOT_PATH}/data/clean/img{idx}.npy')

        if not ((0 <= channel <= img.shape[0]-1) or (channel == -1)):
            print(f'Error: there are only {img.shape[0]} channels to visualize. channel should be a value between 0 and {img.shape[0]-1}.')
            return

        if channel == -1:
            channel = img.shape[0]-1
        
        sample_idx = self.droplet_list[idx]['sample_idx']
        x, y = self.droplet_list[idx]['x'], self.droplet_list[idx]['y']
        sample_name = self.sample_list[sample_idx]['name']
        plt.title(f'Droplet id = {idx}. Label = {str(label)}\nSample name = {sample_name}, at coordinate {x}, {y}')
        plt.imshow(img[channel,:,:])
        plt.show()

    def __len__(self):
        return len(self.labels)
	
    def __getitem__(self, idx):
        img = np.load(f'{config.ROOT_PATH}/data/clean/img{idx}.npy')
        img = resize(img, (img.shape[0], config.IMG_SIZE[0], config.IMG_SIZE[1]), anti_aliasing=False)
        return torch.tensor(np.array([img[img.shape[0]-1, :, :]])), torch.tensor(self.labels[idx, 1], dtype=torch.long)


def load_datasets(
    batch_size = 64, 
):  
    # for now, train dataset = test

    dataset = Data()

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=[int(len(dataset)*0.8),len(dataset) - int(len(dataset)*0.8)], generator=torch.Generator())
    
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # you can use train and test datasets for visualisations
    # call `train_dataset.show_sample(idx, channel=3)` to show an image of one sample
    # call `train_dataset.show_droplet(idx, channel=3)` to show an image of one droplet
    return train_dataset, test_dataset, train_dataloader, test_dataloader
