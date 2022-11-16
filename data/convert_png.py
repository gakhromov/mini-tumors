from PIL import Image
import numpy as np
import cv2 as cv
from tqdm import tqdm
import numpy as np

'''
Automl requires that images be in a recognised image format to work smoothly. For this purpose convert 
all images to png.
'''

def convert(img):
    data = img / img.max() #normalizes data in range 0 - 255
    data = 255 * data
    img = data.astype(np.uint8)
    return img


I_PATH = './augmented/'
O_PATH = './converted/'

labels = np.load(I_PATH + 'labels.npy')

for idx in tqdm(range(len(labels))):
    img = np.load(f'{I_PATH}img{idx}.npy')
    img = img[img.shape[0]-1,:,:]
    cv.imwrite(f'{O_PATH}img{idx}.png', convert(img))

