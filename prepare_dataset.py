import os

import pandas as pd
import numpy as np
from nd2reader import ND2Reader
from skimage.transform import resize
import cv2
import json
from tqdm import tqdm

from data import config


def patch_dataset():
    # Bad code incoming
    # This code is a patch for a specific dataset problem. 
    # You can just ignore this function (and even remove it completely) if your dataset follows the structure
    # specified in the docs/preparing_data.md instructions

    # Fix #1
    # In this particular dataset, there was a problem in the xlsx file: some rows/columns were shifted.
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
    """
    Creates a metadata list of all images with references to xlsx and info files
    """
    sample_list = []
    for dataset in datasets:
        directory = f'{config.ROOT_PATH}/data/TumorScoring/{dataset}/'
        cur_list = sorted(os.listdir(directory + 'ImageFiles/'))
        for fname in cur_list:
            if fname.endswith('.nd2'):
                # extract sample name from the image file name
                f = fname.split('.nd2')[0]
                # open image to get image stats
                sample_img = ND2Reader(f'{directory}ImageFiles/{f}.nd2')
                sample_img.iter_axes = 'c'
                sample_img = np.array(sample_img)
                # compute stats from the image
                stats = []
                for channel in range(sample_img.shape[0]):
                    sample_img_ch = sample_img[channel].flatten()
                    stat = {
                        'max': int(np.max(sample_img_ch)), 'min': int(np.min(sample_img_ch)),
                        'percentile': float(np.percentile(sample_img_ch, config.PERCENTILE)),
                    }
                    stats.append(stat)
                # fill the stats
                suffix = '_No_tt1-tt1_diamRed0_move40'
                sample_list.append({
                    'name': f,
                    'img_path': f'{directory}ImageFiles/{f}.nd2',
                    'xlsx_path': f'{directory}ScoringFiles/{f+suffix}/{f+suffix}.xlsx',
                    'info_txt': f'{directory}ScoringFiles/{f+suffix}/{f+suffix}_Info.txt',
                    'stats': stats,
                })
    return sample_list

def create_splits_with_labels(
    sample_list: list, droplet_list: list, sample_idx: int, labels: list
):
    """
    Iterates through droplets in the sample and creates image files 
    + labels file + metadata for droplets
    """
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
            # check for bad images
            if img_is_good(drop_img, **config.HOUGH_CIRCLES):
                np.save(f'{config.ROOT_PATH}/data/clean/img{clean_idx}.npy', drop_img)
                # label
                label = int(df_labels['label'].loc[droplet])
                labels.append((clean_idx, label))
                # save metadata
                droplet_list.append({'sample_idx': sample_idx, 'x': x, 'y': y, 'r': r})

def img_is_good(img, dp, minDist, param1, param2, minRadius, pct, resize_size):
    """
    Checks if the image has one clear circle in the centre of approx. the size of the image
    @return True if the img has a one clear big circle, False otherwise
    """
    def check_circle(det_circle, true_circle_r, pct):
        '''
        Similarity metric for detected circles and actual droplet
        @param pct -- maximum deviation as fraction of true metrics
        @return True if any of circles are good
        '''
         # Check center
        offset_from_origin = ((true_circle_r-det_circle[0])**2 + (true_circle_r-det_circle[1])**2)**0.5
        check_origin = offset_from_origin <= pct * true_circle_r
        check_r = (1-pct) * true_circle_r <= det_circle[2] <= (1+pct) * true_circle_r
        return check_r and check_origin

    # convert to openCV format
    img_cv = (img / img.max() * 255)[img.shape[0]-1, :, :]
    # resize for faster computation
    img_cv = resize(img_cv, resize_size, anti_aliasing=False)
    img_cv = np.array(img_cv, dtype=np.uint8)
    # detect circles
    circles = cv2.HoughCircles(
        img_cv, cv2.HOUGH_GRADIENT, dp, minDist,
        param1=param1, param2=param2,
        minRadius=minRadius, maxRadius=img_cv.shape[0]//2,
    )
    # get the list of circles
    if circles is None:
        circles = []
    else:
        circles = circles[0]
    
    img_ok = False
    for circle in circles:
        img_ok = img_ok or check_circle(circle, resize_size[0]//2, pct)
    
    return img_ok
    

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
    else:
        print('The dataset has already been prepared. If you want to prepare a new set of datasets, delete the data/clean folder and re-run the script.')

