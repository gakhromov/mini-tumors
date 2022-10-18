import os

import nd2
import matplotlib.pyplot as plt

import config


class Data:
    def __init__(self):
        # create a list with all image full filenames
        self.img_fn_list = []
        for dataset in config.DATASETS:
            directory = f'{os.getcwd()}/data/TumorScoring/{dataset}/ImageFiles/'
            cur_list = os.listdir(directory)
            self.img_fn_list += [directory + f for f in cur_list if f.endswith('.nd2')]
		
        self.img_fn_list = sorted(self.img_fn_list)
	
    def __len__(self):
        return len(self.img_fn_list)
	
    def __getitem__(self, idx):
        img = nd2.imread(self.img_fn_list[idx])
        return img, self.y[idx]
	
    def show_img(self, idx: int, channel: int = 3):
        if idx >= len(self.img_fn_list):
            print(f'Error: index too large (there are {len(self.img_fn_list)} elements)')
            return
        if not (0 <= channel <= 3):
            print(f'Error: there are only 4 channels: (0,1,2 and 3)')
            return

        img = nd2.imread(self.img_fn_list[idx])
        plt.imshow(img[channel,:,:])
        plt.show()
		
		

if __name__ == '__main__':
    data = Data()
    data.show_img(0)
