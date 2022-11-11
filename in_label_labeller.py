'''
Small utility for manually labelling part of dataset for use with VAE oulier detection
'''

from data.data import Data
import numpy as np
from matplotlib import pyplot as plt

# Pick random samples to label
dataset = Data()
no_to_label = 1500
idxs = np.random.randint(1, len(dataset), no_to_label)

for i, img in enumerate(dataset):
    # Display img
    plt.imshow(img)
    # Ask for user input

    # Make decision and go to next instance

