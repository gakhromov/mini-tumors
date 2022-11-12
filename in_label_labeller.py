'''
Small utility for manually labelling part of dataset for use with VAE oulier detection
'''

from data.data import Data
import numpy as np
from matplotlib import pyplot as plt
from augment_data import AugmentedData
import pandas as pd

dataset = Data()

# Pick random samples to label
no_to_label = 600

np.random.seed(1337)
idxs = np.random.randint(1, len(dataset), no_to_label)


plt.ion()

results = pd.DataFrame(columns=['Index', 'Outlier'])

sofar = 0
for i in idxs:
    sofar += 1
    # Display img
    img = dataset.__getitem__(i)
    plt.clf()
    plt.imshow(np.array(img[0][0,:,:]))
    # Ask for user input
    print(f'{sofar} |  Type anything if this data instance is an oulier:')
    inp = input()
    
    # Make decision and go to next instance
    if inp == '':
        print('Not Outlier!')
        results = results.append({'Index':i , 'Outlier':False}, ignore_index=True)
    else:
        print('Outlier!')
        results = results.append({'Index':i , 'Outlier':True}, ignore_index=True)

results.to_csv(f'LabelledOutliers_{no_to_label}.csv')