# Instructions for local training
[<< go back to the main README](../README.md)
1. Make sure that the training dataset is properly set up. All the necessary data resulting from the prepare_dataset.py script should be in the 'data/train' subfolder.
2. Run the training script with `pipenv run python train.py`. If desired, you can also manually specify the number for which to train the model (by default, 15) and the file name under which to save the best model (by default, saved as 'saved_models/best_model.pt'). For example, you could run `pipenv run python main.py --n-epochs 5 --model-filename "saved_models/custom_model_name.pt"`.


# Instructions for training on Euler
[<< go back to the main README](../README.md)
1. Access the Euler cluster via SSH connection. See [this tutorial](https://scicomp.ethz.ch/wiki/Accessing_the_clusters). I recommend using a program that provides some kind of GUI so that you can easily navigate the cluster environment (I personally use MobaXterm). You need to be on the ETH network (directly or via VPN) to access Euler.
2. Type the command `module load gcc/8.2.0 python_gpu/3.10.4`. **NB**: you can avoid having to enter this command everytime you access the cluster by appending those lines (`module load gcc/8.2.0 python_gpu/3.10.4`) to the `.bashrc` file in your home directory on the cluster.
3. In the cluster, navigate to the repo and type a command of the following type:
```
bsub -W 72:00 -R "rusage[mem=2048,ngpus_excl_p=1]" -R "select[gpu_model0=NVIDIAGeForceGTX1080]" -o output_file.txt "pipenv run python train.py --n-epochs 5 --model-filename "saved_models/custom_model_name.pt"
```
For the possible arguments that you can call, please refer to the local training instructions above.
In general, you call bsub, follow it with the arguments of your request, then end with a double-quoted version of the command that you would want to call on your local computer. Here is [documentation](https://scicomp.ethz.ch/wiki/LSF_mini_reference) for bsub commands and their arguments. In the above example command, the used arguments are explained below:
- `-W` : Maximum computation time that you request for (here, 72 hours)
- `-R` : Arguments related to the computational power you want to ask for. n_gpus_excl_p is the number of GPUs requested, mem is the memory requested per GPU (in MB). You can also ask for a specific GPU model if desired with gpu_model0. See [here](https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs) for a list of available GPUs and their LSF specifiers.
- `-o` : the desired name of the output text file that will store every information that would otherwise be printed on the terminal in an equivalent local run (in our case, traning progress, training and validation metrics for each epoch, etc). Note: when a job is running, you can check its current output with the bpeek command.

In the Slurm system, an equivalent command would look as such (NOTE: this has not been tested on our side):
```
sbatch --gpus=1 --gpumem:2g --output="./job_output.txt" --wrap="pipenv run python .py --n_epochs 10 --state 'train'"
```