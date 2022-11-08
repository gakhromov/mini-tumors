import numpy as np
from matplotlib import pyplot as plt
import sys

#plt.imshow(np.load( 'img'+ sys.argv[1] + '.npy'))
#plt.show()

while True:
	no = input()[-1]
	plt.imshow(np.load( 'img'+ no + '.npy'))
	plt.show()
