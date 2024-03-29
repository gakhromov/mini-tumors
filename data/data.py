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
    def __init__(self, transform = None, img_size = -1):
        '''
        transform -- pytorch transform or Composition of several transforms applied to each fetched item.
        '''
        self.labels = np.load(f'{config.ROOT_PATH}/data/train/labels.npy')
        with open(f'{config.ROOT_PATH}/data/train/samples.json', 'r') as f:
            self.sample_list = json.load(f)['samples']
        with open(f'{config.ROOT_PATH}/data/train/droplets.json', 'r') as f:
            self.droplet_list = json.load(f)['droplets']
        
        self.transform = transform
        self.img_size = img_size
	
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
        img = np.load(f'{config.ROOT_PATH}/data/train/img{idx}.npy')

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
        img = np.load(f'{config.ROOT_PATH}/data/train/img{idx}.npy')
        
        # resize
        if self.img_size == -1:
            img = resize(img, (img.shape[0], config.IMG_SIZE[0], config.IMG_SIZE[1]), anti_aliasing=True, preserve_range=True)
        else:
            img = resize(img, (img.shape[0], self.img_size, self.img_size), anti_aliasing=False)

        # take last channel
        channel = img.shape[0]-1
        img = img[channel, :, :]
        # normalise
        sample_idx = self.droplet_list[idx]['sample_idx']
        norm_min = self.sample_list[sample_idx]['stats'][channel]['min']
        norm_max = self.sample_list[sample_idx]['stats'][channel]['percentile']
        img = self.__min_max_norm(img, norm_min, norm_max)

        # transform
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = img[None,:,:]
        return torch.tensor(np.array(img), dtype=torch.float32).repeat(3, 1, 1), torch.tensor(self.labels[idx, 1], dtype=torch.long)
        # return img, torch.tensor(self.labels[idx, 1], dtype=torch.long)
    
    def __min_max_norm(self, x, min=None, max=None, clip=True):
        if min is None:
            min = np.min(x)
        if max is None:
            max = np.max(x)
        x_norm = (x - min) / (max - min)
        if clip:
            x_norm = np.clip(x_norm, 0, 1)
        return x_norm
        

def load_datasets(
    batch_size = 64, 
    img_size = -1,
    use_sampler = False
):  
    # for now, train dataset = test

    dataset = Data(img_size = img_size)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=[int(len(dataset)*0.8),len(dataset) - int(len(dataset)*0.8)], generator=torch.Generator())
    
    if use_sampler:
        sampler_train = sampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler_train)
    else:
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # you can use train and test datasets for visualisations
    # call `train_dataset.show_sample(idx, channel=3)` to show an image of one sample
    # call `train_dataset.show_droplet(idx, channel=3)` to show an image of one droplet
    return train_dataset, test_dataset, train_dataloader, test_dataloader

def sampler(dataset):
    classes = [0,0,0,0]
    for img, label in dataset:
        classes[label] += 1
    class_weights = [len(dataset)/cl for cl in classes]
    weights = [class_weights[label] for img,label in dataset]
    sampler = torch.utils.data.WeightedRandomSampler(weights, min(classes)*4, replacement=True)
    return sampler



class Data_Inference:
    def __init__(self, transform = None, img_size = -1):
        '''
        transform -- pytorch transform or Composition of several transforms applied to each fetched item.
        '''
        with open(f'{config.ROOT_PATH}/data/inference/droplets.json', 'r') as f:
            self.droplet_list = json.load(f)['droplets']
        with open(f'{config.ROOT_PATH}/data/inference/samples.json', 'r') as f:
            self.sample_list = json.load(f)['samples']


        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len([x for x in os.listdir(f'{config.ROOT_PATH}/data/inference') if x[-4:] == '.npy'])
    
    def __getitem__(self, idx):
        img = np.load(f'{config.ROOT_PATH}/data/inference/img{idx}.npy')
        
        # resize
        if self.img_size == -1:
            img = resize(img, (img.shape[0], config.IMG_SIZE[0], config.IMG_SIZE[1]), anti_aliasing=True, preserve_range=True)
        else:
            img = resize(img, (img.shape[0], self.img_size, self.img_size), anti_aliasing=False)

        # take last channel
        channel = img.shape[0]-1
        img = img[channel, :, :]
        # normalise
        sample_idx = self.droplet_list[idx]['sample_idx']
        norm_min = self.sample_list[sample_idx]['stats'][channel]['min']
        norm_max = self.sample_list[sample_idx]['stats'][channel]['percentile']
        img = self.__min_max_norm(img, norm_min, norm_max)

        # transform
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = img[None,:,:]
        return torch.tensor(np.array(img), dtype=torch.float32).repeat(3, 1, 1)
    
    def __min_max_norm(self, x, min=None, max=None, clip=True):
        if min is None:
            min = np.min(x)
        if max is None:
            max = np.max(x)
        x_norm = (x - min) / (max - min)
        if clip:
            x_norm = np.clip(x_norm, 0, 1)
        return x_norm


def load_datasets_inference(
    batch_size = 64, 
    img_size = -1,
):  

    inference_dataset = Data_Inference(img_size = img_size)
    
    inference_dataloader = torch.utils.data.DataLoader(dataset=inference_dataset, batch_size=batch_size, shuffle=False)

    return inference_dataset, inference_dataloader
