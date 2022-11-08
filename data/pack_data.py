'''
Auxiliarry utility for resizing, packing and compressing data for faster uploading to google collab.
'''
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from config import IMG_SIZE


IPATH = './augmented/'
OPATH = './archive/'

labels = np.load(f'{IPATH}/labels.npy')
images = []

print('Gathering images ...')

for idx in tqdm(range(len(labels))):
    images.append( resize(
            np.load(f'{IPATH}/img{idx}.npy'),
            (IMG_SIZE[0], IMG_SIZE[1])))
    #print(len(images))

print('Compressing images ...')

np.savez_compressed(OPATH + 'AugImagesArchive.npz', 
            images = images,
            labels = labels
            )