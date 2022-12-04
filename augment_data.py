'''
Offline augmentation.
Read in clean data from $PROJECT/data/clean/ and write augmented version of dataset to $PROJECT/data/augmented/.
'''

import os
from data import config
import sys
import numpy as np
import cv2
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from torchvision import transforms

from data.data import Data, sampler
from data import filters

import json
from tqdm import tqdm
import torch


class DataAugmentor:
    '''
    Perform offline data augmentation by passing each example through 3 stages of filter.
    '''
    
    def __init__(self, filters_1, filters_2, filters_3):
        '''
        Specify what filters should be applied and in what stages
        
        Apply stage 1 (ProbabilisticFilter / Filter) then stage 2 (MultipleFilter)
        and finally stage 3 (ProbabilisticFilter/ Filter)
        '''
        self.clean_data = Data()
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.filters_3 = filters_3

    def augment_data(self):
        '''
        Read through entire dataset in $PROJECT/data/clean. For each image, remove redundant channels
        and apply augmentation techniques potentially resulting in several images.
        Output them to $PROJECT/data/augmented in same format as cleaned data.
        '''
        # Store lists of new indecies and labels
        self.augmented_labels = []
        self.new_inx = 0
        self.sample_list = []

        with open(f'{config.ROOT_PATH}/data/clean/samples.json', 'r') as f:
            old_sample_list = json.load(f)

        # For each example in dataset
        for ind in tqdm(range(len(self.clean_data.labels))):
            # Read in example
            original_pic, lab = self.clean_data.__getitem__(ind)

            # Shape 1 64 64

            # Normalize / convert to CV2 grayscale image
            # Assumption: Image is floats from 0 to 1
            # original_pic = np.array(np.array(original_pic) * 255, dtype = np.uint8)

            # Apply universal filters to image
            for filter in self.filters_1:
                original_pic = filter(original_pic.float())

            # Create additional copies of image
            pics = [original_pic]
            for filter in self.filters_2:
                pics = pics + filter(original_pic)

            # For each of the several pics outputted by stage 2
            for pic in pics:
                # Apply Random transformations to images
                for filter in self.filters_3:
                    pic = filter(pic)

                # Write out image
                np.save(f'{config.ROOT_PATH}/data/augmented/img{self.new_inx}.npy', pic)
                                
                # Add labels of generated images to labels array
                self.augmented_labels.append(lab)

                # Increment index
                self.new_inx = self.new_inx + 1

        # Record augmented dataset labels
        np.save(f'{config.ROOT_PATH}/data/augmented/labels.npy', np.array(self.augmented_labels, np.int8))


class AugmentedData(Data):
    '''
    Seperate data loader for Augmented data.
    '''
    def __init__(self, transform = None, indecies = None, resize = True):
        '''
        If indecies is None load entire augmented dataset. Else, load subset of the dataset indexed
        by indecies.
        '''
        self.resize = resize
        if indecies is None:
            self.labels = np.load(f'{config.ROOT_PATH}/data/augmented/labels.npy')
            self.indecies = range(len(self.labels))
        else:
            self.labels = np.take(np.load(f'{config.ROOT_PATH}/data/augmented/labels.npy'), indices=indecies)
            self.indecies = indecies

        self.transform = transform

    '''
    Implement __len__ and __getitem__ for compatibility with pytorch
    '''
    def __len__(self):
        return len(self.indecies)

    def __getitem__(self, idx):
        # Get actual index
        ds_index = self.indecies[idx]
        img = np.load(f'{config.ROOT_PATH}/data/augmented/img{ds_index}.npy')
        if self.resize == True:
            #print('Complaint: resizing already done once, impossible to upscale beyond current config.') 
            img = resize(img, (1, config.IMG_SIZE[0], config.IMG_SIZE[1]), anti_aliasing=True, preserve_range=True)
        if self.transform != None:
            img = self.transform(img)
        return torch.tensor(img, dtype=torch.float64), torch.tensor(self.labels[idx], dtype=torch.long)
    

def create_split(batch_size=64, use_sampler=False, test_transform = None, train_transform = None, test_percentage = 0.2):
    '''
    Create and return two AugmentedData objects, one each for the train and test split. Test split is 
    test_percentage sized random sample of entire dataset.
    '''
    all_data = AugmentedData()

    train_idx, test_idx = train_test_split(list(range(len(all_data))), test_size=test_percentage)

    train_data = AugmentedData( train_transform, indecies=train_idx)
    
    test_data = AugmentedData( test_transform, indecies=test_idx)

    if use_sampler:
        sampler_train = sampler(train_data)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, sampler=sampler_train)
    else:
        train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return train_data, test_data, train_dataloader, test_dataloader


if __name__ == '__main__':
    '''
    This is currently what the other team claims to have done. I/E Normalize data and center crop
    # TODO: Normalization
    '''
    
    stage1 = [torch.Tensor, transforms.Lambda(lambda img : torch.cat( [img[:,:,:], img[:,:,:] , img[:,:,:]], dim=0) ) ]# transforms.Normalize(mean, std, inplace=False) ]
    stage2 = [filters.Reflections()]

    stage3 = []
    if not os.path.exists(f'{config.ROOT_PATH}/data/augmented'):
        os.makedirs(f'{config.ROOT_PATH}/data/augmented')
        da = DataAugmentor(stage1, stage2, stage3)
        da.augment_data()
    else:
        print(f'ERROR: {config.ROOT_PATH}/data/augmented already exists. Perhaps the augmented dataset is already generated', file=sys.stderr)
