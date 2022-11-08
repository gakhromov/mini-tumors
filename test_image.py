import matplotlib.pyplot as plt
import numpy as np

img = np.load("/Users/arthur/Documents/ETH/DS_Lab/mini-tumors/data/clean/img70.npy")
plt.imshow(img[3])