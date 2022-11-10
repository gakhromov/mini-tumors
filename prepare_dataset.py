import os

import pandas as pd
import numpy as np
import nd2
from nd2reader import ND2Reader
import json
from tqdm import tqdm

from data import config


def patch_dataset():
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

def get_sample_list(datasets: list):
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

def create_splits_with_labels(
    sample_list: list, droplet_list: list, sample_idx: int, labels: list
):
    df = pd.read_excel(
        sample_list[sample_idx]['xlsx_path'],
        index_col=0,
        sheet_name=None,
    )
    feats = df['Sheet1']

    label_sheet_name = [k for k in df.keys() if k != 'Sheet1']
    # check for absence of labels
    if len(label_sheet_name) != 0:
        df_labels = pd.read_excel(
            sample_list[sample_idx]['xlsx_path'],
            index_col=0,
            header=None,
            names=['label'],
            sheet_name=label_sheet_name[0],
        )
    else:
        # quit if we have no labels
        return

    # img = nd2.imread(sample_list[sample_idx]['img_path'])
    img = ND2Reader(sample_list[sample_idx]['img_path'])
    img.iter_axes = 'c'
    img = np.array(img)
    
    splits = []
    for droplet in feats.index:
        # extract x & y coordinates of the center + diameter
        x, y, d = feats[['TrueCentroidX', 'TrueCentroidY', 'DiameterMeasure']].loc[droplet]
        x, y, d = map(int, [x, y, d])
        # check if we get to the edge of the image:
        r = int(np.min([d//2, x, y, img.shape[1]-x, img.shape[2]-y]))
        # check for missing labels
        if droplet in df_labels.index:
            clean_idx = len(labels)
            # image
            drop_img = img[:, x-r:x+r, y-r:y+r].astype(np.int32)
            np.save(f'{config.ROOT_PATH}/data/clean/img{clean_idx}.npy', drop_img)
            # label
            label = int(df_labels['label'].loc[droplet])
            labels.append((clean_idx, label))
            # save metadata
            droplet_list.append({'sample_idx': sample_idx, 'x': x, 'y': y, 'r': r})
    

if __name__ == '__main__':
    if not os.path.exists(f'{config.ROOT_PATH}/data/clean'):
        patch_dataset()
        os.makedirs(f'{config.ROOT_PATH}/data/clean')

        # check data folder to look for img files and
        # create a list with all sample info: sample name, img path, xlsx path, info txt path
        sample_list = get_sample_list(config.DATASETS)
        droplet_list = []
        labels = []

        for i, sample in enumerate(tqdm(sample_list)):
            create_splits_with_labels(sample_list, droplet_list, i, labels)
        
        np.save(f'{config.ROOT_PATH}/data/clean/labels.npy', np.array(labels, np.int8))
        with open(f'{config.ROOT_PATH}/data/clean/samples.json', 'w') as f:
            json.dump({'samples': sample_list}, f)
        with open(f'{config.ROOT_PATH}/data/clean/droplets.json', 'w') as f:
            json.dump({'droplets': droplet_list}, f)

