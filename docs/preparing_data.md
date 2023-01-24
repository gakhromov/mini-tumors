# Instructions to prepare the datasets
[<< go back to the main README](../README.md)

In order to work with the data, we need to preprocess the datasets to convert them to the right format. However, scripts rely on a specific structure of the datasets as well. Here is the expected structure:
- `data/TumorScoring/<NameOfTheDataset>/` (example: `/data/TumorScoring/ColonCancer/`) should be the root folder of the dataset
- The root folder should contain two folders: 
    - `ImageFiles` with a list of `.nd2` files;
    - `ScoringFiles` with a list of folders named after each `.nd2` file (without `.nd2` at the end). Each folder should contain a `.txt` (with metadata) and a `.xlsx` file (with the coordinates of individual droplets).

When you have prepared the dataset, follow these steps to set it up for the subsequent scripts:

1. Install your dataset folder into the folder `data/TumorScoring/`

    **NB**: If you use Euler, you might want to follow the instructions form [this tutorial](https://scicomp.ethz.ch/wiki/Storage_and_data_transfer). For example, the following command worked for us when we wanted to upload the dataset folder from our local computer to Euler: 
```bash
scp -r TumorScoring mbrassard@euler.ethz.ch:/cluster/home/mbrassard/mini-tumors/data/
```
2. Configure the names of the datasets you want to use in `data/config.py` in the array `DATASETS`. For example, if you want to use `BrainCancer` and `ColonCancer` datasets, you should type:
```python
DATASETS = [
    "BrainCancer", "ColonCancer"
]
```
2. Run `pipenv run python prepare_dataset.py` to prepare the dataset for use. This script will go through the all the datasets and create `img<INDEX>.npy` files for each droplet, as well as create a `droplets.json` and `samples.json` metadata files. These files should not be modified and are just used for our scripts.