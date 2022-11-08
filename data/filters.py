'''
Skimage and OpenCV do not seem to have a neat way to make pipelines of filters for offline augmentation, 
so i've written a few wrappers that allow this. Basically filters are applied in three stages, first basic 
filters are applied to each image, then a set of filters that output more
than one image and finally aother set of basic filters to each of these.
                                                                                ------> Image 3.1 ---> More filters ---> 4.1
                                                                              /
Image1 ------> Basic Filters ------> Image 2 -----> Multiplying Filters -----              ...
                                                                              \
                                                                                ------> Image 3.n ----> More filters ---> 4.n

Theres a seperate wrapper for probabilistic filters. These are applied randomly.

'''

from abc import ABC, abstractmethod
import random
import cv2
import numpy as np

class Filter(ABC):
    '''
    Abstract parent class for data augmentation filter.
    '''
    @abstractmethod
    def transform(self, img):
        '''
        In child class this should take a grayscale numpy array and 
        return a grayscale numpy array
        '''
        pass

class MultipleFilter(Filter):
    @abstractmethod
    def transform(self, img):
        '''
        In child class this should take a grayscale numpy array and 
        return a LIST OF grayscale numpy arrays
        '''
        pass


class ProbabilisticFilter(Filter):
    probability = None
    def __init__(self):
        if self.probability == None:
            raise NotImplementedError('Subclasses must define probability of transformation occuring.')

    @abstractmethod
    def transform(self, img):
        '''
        In child class this should take a grayscale numpy array and 
        return a grayscale numpy array with the transformation applied
        according to probability
        '''
        pass

############ Basic Filters ###############

class AdaptiveHistogramEqualization(Filter):
    def __init__(self, clim = 2.0, tile_grid_size = (8,8)) -> None:
        super().__init__()
        self.clahe = cv2.createCLAHE(clipLimit=clim, tileGridSize=tile_grid_size)

    def transform(self, img):
        return self.clahe.apply(img)


class HistogramEqualization(Filter):
    def transform(self, img):
        return cv2.equalizeHist(img)

############### Multiple Filters ###############

class Reflections(MultipleFilter):
    '''
    Return the remaining 3 90deg rotations of img
    '''
    def transform(self, img):
        return [np.rot90(img, k=deg) for deg in [1,2,3]]

############### Probabilistic Filters ###############

class BlurFilter(ProbabilisticFilter):
    def __init__(self, prob = 0.2, inten = (1, 8)):
        '''
        prob --> probability that transformation will be applyed
        intensity --> intentisy of transformation chosen uniformly over this interval
        in this case this represents the kernel size of the blur
        '''
        self.probability = prob
        self.intensity = inten
        assert(type(inten[0]) is int)
        assert(type(inten[1]) is int)

    def transform(self, img):
        if random.uniform(0, 1) < self.probability:
            return img
        else:
            self.ks = random.randint(self.intensity[0], self.intensity[1])
            return cv2.blur(img, (self.ks, self.ks))


############### Dict of Implemented Filters ###############

stage1 = [AdaptiveHistogramEqualization()]
stage2 = [Reflections()]
stage3 = [BlurFilter()]

