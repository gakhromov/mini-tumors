import os

import numpy as np
import pandas as pd
import torch
import nd2
import json
from skimage.transform import resize

import matplotlib.pyplot as plt

from data import config


class Data:
    def __init__(self):
        self.labels = np.load(f'{config.ROOT_PATH}/data/clean/labels.npy')
        with open(f'{config.ROOT_PATH}/data/clean/samples.json', 'r') as f:
            self.sample_list = json.load(f)
	
    def show_sample(self, sample_idx: int, channel: int = 3):
        if sample_idx >= len(self.sample_list):
            print(f'Error: index too large (there are {len(self.sample_list)} samples)')
            return
        if not (0 <= channel <= 3):
            print(f'Error: there are only 4 channels: (0,1,2 and 3)')
            return

        img = nd2.imread(self.sample_list[sample_idx]['img_path'])
        plt.title('Sample = ' + self.sample_list[sample_idx]['name'])
        plt.imshow(img[channel,:,:])
        plt.show()

    def show_droplet(self, idx: int, channel: int = 3):
        if idx >= len(self.labels):
            print(f'Error: index too large (there are {len(self.labels)} labeled droplets)')
            return
        if not (0 <= channel <= 3):
            print(f'Error: there are only 4 channels: (0,1,2 and 3)')
            return
        
        img, label = self.__getitem__(idx)
        sample_idx, split_idx = self.global_index_map[idx]
        dropidx, _, _, _, _ = self.sample_list[sample_idx]['splits'][split_idx]
        plt.title(f'Sample = {self.sample_list[sample_idx]["name"]}, DropIdx = {dropidx}. Label = {str(label)}')
        plt.imshow(img[channel,:,:])
        plt.show()

    def __len__(self):
        return len(self.labels)
	
    def __getitem__(self, idx):
        img = np.load(f'{config.ROOT_PATH}/data/clean/img{idx}.npy')
        img = resize(img, (img.shape[0], config.IMG_SIZE[0], config.IMG_SIZE[1]), anti_aliasing=False)
        return img[0, :, :], self.labels[idx, 1]


def load_datasets(
    batch_size = 64, 
):  
    # for now, train dataset = test
    train_dataset = Data()
    test_dataset = Data()
    
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # you can use train and test datasets for visualisations
    # call `train_dataset.show_sample(idx, channel=3)` to show an image of one sample
    # call `train_dataset.show_droplet(idx, channel=3)` to show an image of one droplet
    return train_dataset, test_dataset, train_dataloader, test_dataloader
