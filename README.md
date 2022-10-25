# mini-tumors
Data Science Lab 2022 Project - Mini-Tumors in Nanoliter-Droplets for Personalized Cancer Treatment

## Setup (local)
1. Install Python 3.10.4 via [pyenv](https://github.com/pyenv/pyenv) or the [offical website](https://www.python.org/downloads/)
2. Install [pipenv](https://github.com/pypa/pipenv) using `pip install pipenv` for package version management
3. Install necessary packages by `pipenv install` (`pipenv install --dev` to install additional dev packages)
4. Install the dataset folder `TumorScoring` into the folder `/data`
5. Run `pipenv run python prepare_dataset.py` to prepare the dataset for use

## Setup (Euler)
1. Type the command `module load python_gpu/3.10.4`
2. Install the dataset folder `TumorScoring` into the folder `/data`
3. ?

## Running
1. ?

## Packages
All packages are listed in the `Pipfile`. Please install new package versions in accordance to this page [Python on Euler](https://scicomp.ethz.ch/wiki/Python_on_Euler), so that the package versions are synced between Euler and your local machine.
