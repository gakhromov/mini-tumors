# mini-tumors
Data Science Lab 2022 Project - Mini-Tumors in Nanoliter-Droplets for Personalized Cancer Treatment

## Setup (local)
1. Install Python 3.10.4 via [pyenv](https://github.com/pyenv/pyenv) or the [offical website](https://www.python.org/downloads/)
2. Install [pipenv](https://github.com/pypa/pipenv) using `pip install pipenv` for package version management
3. Install necessary packages by `pipenv install` (`pipenv install --dev` to install additional dev packages)
4. Install the dataset folder `TumorScoring` into the folder `/data`
5. Run `pipenv run python prepare_dataset.py` to prepare the dataset for use

## Running (local)
1. ?

## Setup (Euler)
1. Access the Euler cluster via SSH connection. See [this tutorial](https://scicomp.ethz.ch/wiki/Accessing_the_clusters). I recommend using a program that provides some kind of GUI so that you can easily navigate the cluster environment (I personally use MobaXterm). You need to be on the ETH network (directly or via VPN) to access Euler.
2. Type the command `module load gcc/8.2.0 python_gpu/3.10.4`.
3. Clone this github repo into your Euler home directory using classic Git commands such as git clone.
4. In the cluster, navigate to the repo and type the following commands:
```
pip install pipenv
pipenv install --dev
```
5. Upload the dataset folder `TumorScoring` from your local computer into the subfolder `mini-tumors/data` on the Euler cluster. This is easily done with a GUI; if you want to use command line, you are going to use scp commands (see [this tutorial](https://scicomp.ethz.ch/wiki/Storage_and_data_transfer)). For example, it worked for me using this command on my local computer command line: 
```
scp -r TumorScoring mbrassard@euler.ethz.ch:/cluster/home/mbrassard/mini-tumors/data/
```
6. In the repo, run `pipenv run python prepare_dataset.py` to prepare the dataset for use.

## Running (Euler)
1. Access the Euler cluster via SSH connection. See [this tutorial](https://scicomp.ethz.ch/wiki/Accessing_the_clusters). I recommend using a program that provides some kind of GUI so that you can easily navigate the cluster environment (I personally use MobaXterm). You need to be on the ETH network (directly or via VPN) to access Euler.
2. Type the command `module load gcc/8.2.0 python_gpu/3.10.4`. NOTE: you can avoid having to enter this command everytime you access the cluster by appending those lines (`module load gcc/8.2.0 python_gpu/3.10.4`) to the `.bashrc` file in your home directory on the cluster.
3. In the cluster, navigate to the repo and type the following command (TEMPORARY COMMAND THAT PROBABLY DOESN'T WORK, WILL BE CHANGED)
```
sbatch --gpus=1 --gpumem:8g --output="./job_output.txt" --wrap="pipenv run python main.py --n_epochs 10 --state 'train'"
```

## Packages
All packages are listed in the `Pipfile`. Please install new package versions in accordance to this page [Python on Euler](https://scicomp.ethz.ch/wiki/Python_on_Euler), so that the package versions are synced between Euler and your local machine.