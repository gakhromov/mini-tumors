import os

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import json
import nd2

from data import config


class Data:
    def __init__(self):
        self.patch_dataset()

        if os.path.exists(f'{config.ROOT_PATH}/data/dataset_meta.json'):
            with open(f'{config.ROOT_PATH}/data/dataset_meta.json', 'r') as f:
                data = json.load(f)
                self.sample_list = data['sample_list']
                self.global_index_map = data['global_index_map']
        else:
            # check data folder to look for img files and
            # create a list with all sample info: sample name, img path, xlsx path, info txt path
            self.sample_list = self.get_sample_list(config.DATASETS)
            # create a global index 0:num droplets in all images
            # each element will point to a specific sample indx (img file) and split (droplet) id there
            self.global_index_map = []

            for i, sample in enumerate(self.sample_list):
                sample['splits'] = self.create_splits_with_labels(i)
                self.global_index_map += list(map(lambda x: [i, x], range(len(sample['splits']))))
            
            with open(f'{config.ROOT_PATH}/data/dataset_meta.json', 'w') as f:
                json.dump({
                    'sample_list': self.sample_list, 
                    'global_index_map': self.global_index_map,
                    }, f)
                print('Meta data (dataset_meta.json) successfully generated')

    def get_sample_list(self, datasets: list):
        sample_list = []
        for dataset in datasets:
            directory = f'{config.ROOT_PATH}/data/TumorScoring/{dataset}/'
            cur_list = sorted(os.listdir(directory + 'ImageFiles/'))
            for fname in cur_list:
                if fname.endswith('.nd2'):
                    # extract sample name from the image file name
                    f = fname.split('.nd2')[0]
                    suffix = '_No_tt1-tt1_diamRed0_move40'
                    sample_list.append({
                        'name': f,
                        'img_path': f'{directory}ImageFiles/{f}.nd2',
                        'xlsx_path': f'{directory}ScoringFiles/{f+suffix}/{f+suffix}.xlsx',
                        'info_txt': f'{directory}ScoringFiles/{f+suffix}/{f+suffix}_Info.txt',
                    })
        return sample_list
    
    def create_splits_with_labels(self, sample_idx: int):
        df = pd.read_excel(
            self.sample_list[sample_idx]['xlsx_path'],
            index_col=0,
            sheet_name=None,
        )
        feats = df['Sheet1']

        label_sheet_name = [k for k in df.keys() if k != 'Sheet1']
        # check for absence of labels
        if len(label_sheet_name) != 0:
            labels = pd.read_excel(
                self.sample_list[sample_idx]['xlsx_path'],
                index_col=0,
                header=None,
                names=['label'],
                sheet_name=label_sheet_name[0],
            )
        else:
            labels = None
        img = nd2.imread(self.sample_list[sample_idx]['img_path'])
        
        splits = []
        for droplet in feats.index:
            # extract x & y coordinates of the center + diameter
            x, y, d = feats[['TrueCentroidX', 'TrueCentroidY', 'DiameterMeasure']].loc[droplet]
            x, y, d = map(int, [x, y, d])
            # compute the radius (so that we don't have negative values)
            r = int(np.min([d//2, x, y, img.shape[1]-x, img.shape[2]-y]))
            # check for missing labels
            if (labels is not None) and (droplet in labels.index):
                splits.append([int(droplet), x, y, r, int(labels['label'].loc[droplet])])
            else:
                splits.append([int(droplet), x, y, r, None])
        
        # how to split the image:
        # img[channel, x-r:x+r, y-r:y+r]
        return splits
	
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
        if idx >= len(self.global_index_map):
            print(f'Error: index too large (there are {len(self.global_index_map)} droplets)')
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
    
    def patch_dataset(self):
        # Bad code incoming
        # thnx for a clean dataset 👍
        # Fix #1
        f = '20220722PM_SulfoB1_T1_Split1'
        suffix = '_No_tt1-tt1_diamRed0_move40'
        if not os.path.exists(f'{config.ROOT_PATH}/data/TumorScoring/ColonCancer/ScoringFiles/{f+suffix}/{f+suffix}_old.xlsx'):
            df = pd.read_excel(
                f'{config.ROOT_PATH}/data/TumorScoring/ColonCancer/ScoringFiles/{f+suffix}/{f+suffix}.xlsx',
                index_col=0,
                sheet_name='0812_1020_sorting',
                names=['label'],
                header=None,
            )
            # shift all col elements by one upwards
            for id in range(len(df.index)):
                if id == len(df.index)-1:
                    continue
                df['label'].iloc[id] = df['label'].iloc[id+1]
            # remove las NaN element
            df.drop(df.tail(1).index, inplace=True)

            df_sheet1 = pd.read_excel(
                f'{config.ROOT_PATH}/data/TumorScoring/ColonCancer/ScoringFiles/{f+suffix}/{f+suffix}.xlsx',
                index_col=0,
                sheet_name='Sheet1',
            )
            # rename old file
            os.rename(
                f'{config.ROOT_PATH}/data/TumorScoring/ColonCancer/ScoringFiles/{f+suffix}/{f+suffix}.xlsx',
                f'{config.ROOT_PATH}/data/TumorScoring/ColonCancer/ScoringFiles/{f+suffix}/{f+suffix}_old.xlsx'
            )
            # save corrected file
            with pd.ExcelWriter(f'{config.ROOT_PATH}/data/TumorScoring/ColonCancer/ScoringFiles/{f+suffix}/{f+suffix}.xlsx') as writer:
                df_sheet1.to_excel(writer, sheet_name='Sheet1', index=True)
                df.to_excel(writer, sheet_name='0812_1020_sorting', index=True, header=None)

    def __len__(self):
        return len(self.global_index_map)
	
    def __getitem__(self, idx):
        sample_idx, split_idx = self.global_index_map[idx]
        img = nd2.imread(self.sample_list[sample_idx]['img_path'])
        _, x, y, r, label = self.sample_list[sample_idx]['splits'][split_idx]
        return img[:, x-r:x+r, y-r:y+r], label


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
