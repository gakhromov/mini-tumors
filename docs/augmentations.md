# Instructions to setup Data Augmentation
## Offline Data Augmentation
[<< go back to the main README](../README.md)

To Run offline data augmentation and generate an augmented/transformed version of the dataset:

1. Specify what transformations you would like to perform on each image in `~/data/samples` accoding to the figure below. Filters are performed as part of one of three stages. Filters in stage 1 and 3 are one to one mappings of input images to output. Filters in stage two map a single input image to one or mores. output images, each perturbed slightly differently.

![Example Transformation Stages](https://i.imgur.com/JXZXueG.png)

Specify what transformations you would like to perform on each image in `~/data/samples`. You can specify transformations by adding them to the `__main__()` function in `augment_data.py` in the appropriate variables. All filters/transformations
specified should be callable objects that accept a single argument (input image) and return either a single image (stage 1 and 3) or a list of images (stage 2). If you would like to write your own transformation you can inherit the abstract classes specified in `data/filters.py`. If one of the lists of stages is left empty, it will perform the identiy transformation.

PS: If this feels overengineered, just leave stage2 and stage3 as emtpy lists and run the code.

2. Augmented data can now be loaded with the `AugmentedData` data class `agument_data.py`. It's constructor takes two optional parameters. The first called `labels` specifies a subset of the complete datasets indecies that the loader should load. This can be used to make test/train splits by providiing it with a random subsample of all indecies. The second specifies a transformation or composition of transformations for online data augmentation. These will be applied on each call to the `__getitem__` method. 
