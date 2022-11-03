'''
Offline augmentation.
Read in clean data from $PROJECT/data/clean/ and write augmented version of dataset to $PROJECT/data/augmented/.
'''
import os
from data import config
import sys
import numpy as np
import cv2

from data.data import Data
from data import filters

from tqdm import tqdm

class DataAugmentor:
    '''
    TODO: Doccument
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

        # For each example in dataset
        for ind in tqdm(range(len(self.clean_data.labels))):
            # Read in example
            original_pic, lab = self.clean_data.__getitem__(ind)

            # Normalize / convert to CV2 grayscale image
            # Assumption: Image is floats from 0 to 1
            original_pic = np.array(original_pic * 255, dtype = np.uint8)

            # Apply universal filters to image
            for filter in self.filters_1:
                original_pic = filter.transform(original_pic)

            # Create additional copies of image
            pics = [original_pic]
            for filter in self.filters_2:
                pics = pics + filter.transform(original_pic)


            for pic in pics:
                # Apply Random transformations to images
                for filter in self.filters_3:
                    pic = filter.transform(pic)

                # Write out image
                np.save(f'{config.ROOT_PATH}/data/augmented/img{self.new_inx}.npy', pic)
                                
                # Add labels of generated images to labels array
                self.augmented_labels.append(lab)

                # Increment index
                self.new_inx = self.new_inx + 1

        # Record augmented dataset labels
        np.save(f'{config.ROOT_PATH}/data/augmented/labels.npy', np.array(self.augmented_labels, np.int8))
    

if __name__ == '__main__':
    if not os.path.exists(f'{config.ROOT_PATH}/data/augmented'):
        os.makedirs(f'{config.ROOT_PATH}/data/augmented')
        da = DataAugmentor(filters.stage1, filters.stage2, filters.stage3)
        da.augment_data()
    else:
        print(f'ERROR: {config.ROOT_PATH}/data/augmented already exists. Perhaps the augmented dataset is already generated', file=sys.stderr)
