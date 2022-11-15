# mini-tumors
Data Science Lab 2022 Project - Mini-Tumors in Nanoliter-Droplets for Personalized Cancer Treatment

## Setup (local)
1. Install Python 3.10.4 via [pyenv](https://github.com/pyenv/pyenv) or the [offical website](https://www.python.org/downloads/)
2. Install [pipenv](https://github.com/pypa/pipenv) using `pip install pipenv` for package version management
3. Install necessary packages by `pipenv install` (`pipenv install --dev` to install additional dev packages)
4. Install the dataset folder `TumorScoring` into the folder `/data`
5. Run `pipenv run python prepare_dataset.py` to prepare the dataset for use

## Setup (Data Augmentation)
### Offline Data Augmentation
To Run offline data augmentation and generate an augmented/transformed version of the dataset:

1. Specify what transformations you would like to perform on each image in `~/data/samples` accoding to the figure below. Filters are performed as part of one of three stages. Filters in stage 1 and 3 are one to one mappings of input images to output. Filters in stage two map a single input image to one or mores. output images, each perturbed slightly differently.

![Example Transformation Stages](https://i.imgur.com/JXZXueG.png)

Specify what transformations you would like to perform on each image in `~/data/samples`. You can specify transformations by adding them to the `__main__()` function in `augment_data.py` in the appropriate variables. All filters/transformations
specified should be callable objects that accept a single argument (input image) and return either a single image (stage 1 and 3) or a list of images (stage 2). If you would like to write your own transformation you can inherit the abstract classes specified in `data/filters.py`. If one of the lists of stages is left empty, it will perform the identiy transformation.

PS: If this feels overengineered, just leave stage2 and stage3 as emtpy lists and run the code.

2. Augmented data can now be loaded with the `AugmentedData` data class `agument_data.py`. It's constructor takes two optional parameters. The first called `labels` specifies a subset of the complete datasets indecies that the loader should load. This can be used to make test/train splits by providiing it with a random subsample of all indecies. The second specifies a transformation or composition of transformations for online data augmentation. These will be applied on each call to the `__getitem__` method. 


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
3. In the cluster, navigate to the repo and type a command of the following type:
```
bsub -W 72:00 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0=NVIDIAGeForceGTX1080]" -o output_file.txt "pipenv run python main.py --n_epochs 10 --state 'train'"
```
In general, you call bsub, follow it with the arguments of your request, then end with a double-quoted version of the command that you would want to call on your local computer. Here is [documentation](https://scicomp.ethz.ch/wiki/LSF_mini_reference) for bsub commands and their arguments. In the above example command, the used arguments are explained below:
- -W : Maximum computation time that you request for (here, 72 hours)
- -R : Arguments related to the computational power you want to ask for. n_gpus_excl_p is the number of GPUs requested, mem is the memory requested per GPU (in MB). You can also ask for a specific GPU model if desired with gpu_model0. See [here](https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs) for a list of available GPUs and their LSF specifiers.
- -o : the desired name of the output text file that will store every information that would otherwise be printed on the terminal in an equivalent local run (in our case, traning progress, training and validation metrics for each epoch, etc). Note: when a job is running, you can check its current output with the bpeek command.

In the Slurm system, an equivalent command would look as such (NOTE: so far, this doesn't seem to work):
```
sbatch --gpus=1 --gpumem:2g --output="./job_output.txt" --wrap="pipenv run python .py --n_epochs 10 --state 'train'"
```

## Packages
All packages are listed in the `Pipfile`. Please install new package versions in accordance to this page [Python on Euler](https://scicomp.ethz.ch/wiki/Python_on_Euler), so that the package versions are synced between Euler and your local machine.
